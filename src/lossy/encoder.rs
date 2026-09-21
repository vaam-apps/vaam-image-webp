use std::io::Write;

use byteorder_lite::{LittleEndian, WriteBytesExt};

use super::arithmetic_encoder::ArithmeticEncoder;
use super::common::*;
use super::prediction::*;
use super::transform;
use super::yuv::convert_image_y;
use super::yuv::convert_image_yuv;
use super::Frame;
use crate::ColorType;
use crate::EncodingError;

// currently in decoder it actually stores this information on the macroblock but that's confusing
// because it doesn't update the macroblock, just the complexity values as we decode
// this is used as the complexity per 13.3 in the decoder
#[derive(Clone, Copy, Default)]
struct Complexity {
    y2: u8,
    y: [u8; 4],
    u: [u8; 2],
    v: [u8; 2],
}

impl Complexity {
    fn clear(&mut self, include_y2: bool) {
        self.y = [0; 4];
        self.u = [0; 2];
        self.v = [0; 2];
        if include_y2 {
            self.y2 = 0;
        }
    }
}

#[derive(Default)]
struct QuantizationIndices {
    yac_abs: u8,
    ydc_delta: Option<i8>,
    y2dc_delta: Option<i8>,
    y2ac_delta: Option<i8>,
    uvdc_delta: Option<i8>,
    uvac_delta: Option<i8>,
}

/// TODO: Consider merging this with the MacroBlock from the decoder
#[derive(Clone, Copy, Default)]
struct MacroblockInfo {
    luma_mode: LumaMode,
    // note ideally this would be on LumaMode::B
    // since that it's where it's valid but need to change the decoder to
    // work with that as well
    luma_bpred: Option<[IntraMode; 16]>,
    chroma_mode: ChromaMode,
    // whether the macroblock uses custom segment values
    // if None, will use the frame level values
    segment_id: Option<usize>,

    coeffs_skipped: bool,
}

struct Luma16x16Coeffs {
    y2_coeffs: [i32; 16],
    y_coeffs: LumaYCoeffs,
}

type LumaYCoeffs = [i32; 16 * 16];

type ChromaCoeffs = [i32; 16 * 4];

// --- Rate-distortion mode decision ------------------------------------------
//
// `choose_macroblock_info` picks, for each macroblock, the 16x16 luma
// prediction mode and chroma prediction mode that minimise a Lagrangian cost
// `distortion + lambda * rate`, instead of the fixed DC/DC choice the encoder
// used to make unconditionally. See `choose_macroblock_info` for the search
// itself and `mode_decision_lambda` for how `lambda` is derived.

/// Cheap proxy for the number of bits a block of quantized coefficients will
/// cost to entropy-code, used as the rate term in RD mode decision.
///
/// A full model would run the actual token tree (`DCT_TOKEN_TREE`) and sum
/// `-log2(prob)` per bit, which is expensive to do for every candidate mode
/// of every macroblock. Instead we count, per nonzero coefficient, 1 (for the
/// token-tree traversal that signals "nonzero", which costs roughly a
/// constant number of bits regardless of magnitude) plus the coefficient's
/// absolute level (because larger levels need more extra bits: literals 1-4
/// cost a handful of tree bits, but categories DCT_CAT1..6 add 1-11 explicit
/// sign/magnitude bits whose count grows with `log2(level)` - a linear
/// over-estimate in `level`, but one that preserves the right ordering
/// between "one big coefficient" and "several small ones", which is what
/// mode decision actually needs). This tracks real token cost closely enough
/// in practice (see the rd_eval numbers in the commit message) without
/// needing the token tree at all.
fn coeff_rate_estimate(coeffs: &[i32]) -> i64 {
    coeffs
        .iter()
        .filter(|&&c| c != 0)
        .map(|&c| 1 + i64::from(c.abs()))
        .sum()
}

/// Real entropy cost, in bits, of encoding `value` with a tree-structured
/// binary code and per-node probabilities - the same tree walk
/// `ArithmeticEncoder::write_with_tree` (`write_with_tree_start_index` with
/// `start_index = 0`) performs when it actually writes the symbol, except
/// this computes `-log2(P(branch))` at each node instead of emitting a bit.
///
/// `probabilities[i]` is, per VP8 convention (9.2), the probability (scaled
/// to 0..=255) that the branch at node `i` is *false*; a `false` branch
/// therefore costs `-log2(prob / 256)` and a `true` branch costs
/// `-log2(1 - prob / 256)`.
///
/// This is only used for B_PRED submode rate (see `bpred_mode_bit_cost`
/// below), not for coefficients: `coeff_rate_estimate` is deliberately a
/// cheap magnitude-based proxy because it runs per coefficient, per
/// candidate mode, per macroblock, and getting the exact token-tree cost
/// right matters less than getting *an* ordering right. The B_PRED submode
/// tree is walked at most 10 times per sub-block (one per candidate), and
/// its context tables (`KEYFRAME_BPRED_MODE_PROBS`) swing the real cost by
/// several bits depending on the neighbouring submodes - which is exactly
/// the signal the sub-block search needs in order to prefer, e.g., a
/// slightly-worse-predicting mode that matches its neighbours' mode over a
/// slightly-better one that doesn't. So this pays for the exact
/// probability-weighted cost instead of reusing the coefficient proxy.
///
/// Note this makes `trial_luma_bpred`'s `rate` a mix of two different
/// scales - `coeff_rate_estimate`'s inflated magnitude proxy plus this
/// function's real Shannon bits - added together and then multiplied by
/// the same `lambda` (which `LAMBDA_SCALE` calibrates against the proxy's
/// scale, not bits). This is a known approximation; see
/// `trial_luma_bpred`'s doc comment.
fn tree_bit_cost(tree: &[i8], probabilities: &[Prob], value: i8) -> f64 {
    let mut current_index = tree
        .iter()
        .position(|&x| x == -value)
        .expect("value must be a leaf of this tree");

    let mut bits = 0.0f64;
    loop {
        // Root children live at indices 0 (false) and 1 (true); reaching
        // either means we've walked all the way back and the branch taken
        // to get from that child to `value`'s leaf is not needed anymore
        // - this mirrors `write_with_tree_start_index`'s two `if
        // current_index == start_index (+1)` base cases with
        // `start_index = 0`.
        if current_index == 0 {
            bits += branch_bit_cost(probabilities[0], false);
            break;
        }
        if current_index == 1 {
            bits += branch_bit_cost(probabilities[0], true);
            break;
        }

        // Even index => `current_index` is some node's `false` child;
        // odd => it's that node's `true` child, at `current_index - 1`.
        let (branch_is_true, node_index) = if current_index % 2 == 0 {
            (false, current_index)
        } else {
            (true, current_index - 1)
        };
        bits += branch_bit_cost(probabilities[node_index / 2], branch_is_true);

        // Find the parent: the node whose child pointer equals this node's
        // own index.
        current_index = tree
            .iter()
            .position(|&x| x == node_index as i8)
            .expect("every non-root tree node is some other node's child");
    }
    bits
}

fn branch_bit_cost(prob: Prob, branch_is_true: bool) -> f64 {
    let p_false = f64::from(prob) / 256.0;
    let p = if branch_is_true { 1.0 - p_false } else { p_false };
    // `prob` is a u8 in 0..=255 so `p` never reaches exactly 0 or 1, but
    // guard the log anyway rather than relying on that.
    -p.clamp(f64::MIN_POSITIVE, 1.0).log2()
}

/// Rate, in bits, of signalling one B_PRED sub-block's `IntraMode` given
/// its above (`top`) and left (`left`) neighbouring submodes - the same
/// context `write_macroblock_header` looks up in `KEYFRAME_BPRED_MODE_PROBS`
/// when it actually writes the mode. See `tree_bit_cost` for the mechanism
/// and `trial_luma_bpred` for how this is combined with coefficient rate.
fn bpred_mode_bit_cost(top: IntraMode, left: IntraMode, mode: IntraMode) -> f64 {
    let probs = &KEYFRAME_BPRED_MODE_PROBS[top as usize][left as usize];
    tree_bit_cost(&KEYFRAME_BPRED_MODE_TREE, probs, mode as i8)
}

// Empirically tuned multiplier for `mode_decision_lambda`, see there for the
// derivation. Calibrated against `examples/rd_eval` on the Kodak corpus.
const LAMBDA_SCALE: f64 = 0.02;

/// Lagrangian multiplier for RD mode decision: `cost = distortion + lambda *
/// rate`.
///
/// Distortion here is sum-of-squared-error (SSE) in the pixel domain, and
/// rate is `coeff_rate_estimate`'s coefficient-count-based proxy. Classical
/// rate-distortion theory (Sullivan & Wiegand, "Rate-Distortion Optimization
/// for Video Compression", IEEE Signal Processing Magazine, 1998) derives,
/// for SSD-based distortion against a real bit-rate cost, `lambda = c *
/// Qstep^2`: the quantizer step controls both how much squared error a given
/// residual leaves (variance ~ Qstep^2/12 per coefficient) and how many bits
/// a one-step change in level costs, so both distortion and marginal rate
/// scale with the same step, and lambda (their ratio at the RD-optimal point)
/// scales with its square.
///
/// We use the macroblock's luma AC quantizer step (`segment.yac`) as
/// `Qstep`, since AC coefficients dominate both the coefficient count and
/// the typical energy in a block, and this crate does not yet do
/// per-macroblock quantizer search (segments are unused beyond segment 0),
/// so it is the one step size that is actually representative here.
///
/// The classical bit-domain constant (`c ~= 0.85`) assumes rate is measured
/// in real bits; our `coeff_rate_estimate` proxy is in "coefficient units"
/// (roughly `nonzero_count + sum(|level|)`), which is systematically larger
/// than the real bit cost for the same block (see its doc comment), so the
/// same distortion/rate tradeoff needs a smaller multiplier. `LAMBDA_SCALE`
/// is that multiplier, picked empirically by sweeping it against
/// `examples/rd_eval` on the Kodak corpus and taking the value that
/// minimised the median ours/libwebp size ratio - not re-derived from first
/// principles, since the proxy itself is heuristic.
fn mode_decision_lambda(segment: &Segment) -> f64 {
    let qstep = f64::from(segment.yac);
    LAMBDA_SCALE * qstep * qstep
}

/// `prob_skip_false` (9.10/19.2 in the spec) is the probability, scaled to a
/// byte, that a macroblock's `coeffs_skipped` flag is *false* - i.e. that it
/// is NOT skipped. It only affects how many bits the flag itself costs, not
/// correctness (the arithmetic coder is exact for any probability in
/// 1..=255), so getting it close to the real skip rate is a compression
/// efficiency question, not a correctness one. We measure the real rate via
/// `Vp8Encoder::count_skipped_macroblocks` rather than guessing, since mode
/// decision has to run once for real regardless and re-running it once more
/// dry is cheap by comparison.
fn skip_probability(skipped: u32, total: u32) -> u8 {
    if total == 0 {
        return 128;
    }
    let not_skipped = u64::from(total - skipped);
    // round-to-nearest rather than truncating, and keep clear of the 0/256
    // edges: a probability of exactly 0 (or a raw 256) isn't representable
    // in a u8, and either extreme just means "one branch never happens
    // here", which 1 / 255 already expresses for all practical purposes.
    let scaled = (256 * not_skipped + u64::from(total) / 2) / u64::from(total);
    scaled.clamp(1, 255) as u8
}

struct Vp8Encoder<W> {
    writer: W,
    frame: Frame,
    /// The encoder for the macroblock headers and the compressed frame header
    encoder: ArithmeticEncoder,
    segments: [Segment; MAX_SEGMENTS],
    segments_enabled: bool,

    loop_filter_adjustments: bool,
    macroblock_no_skip_coeff: Option<u8>,
    quantization_indices: QuantizationIndices,

    token_probs: TokenProbTables,

    top_complexity: Vec<Complexity>,
    left_complexity: Complexity,

    top_b_pred: Vec<IntraMode>,
    left_b_pred: [IntraMode; 4],

    macroblock_width: u16,
    macroblock_height: u16,

    /// Partitions of encoders for the macroblock coefficient data
    partitions: Vec<ArithmeticEncoder>,

    // the left borders used in prediction
    left_border_y: [u8; 16 + 1],
    left_border_u: [u8; 8 + 1],
    left_border_v: [u8; 8 + 1],

    // the top borders used in prediction
    top_border_y: Vec<u8>,
    top_border_u: Vec<u8>,
    top_border_v: Vec<u8>,
}

impl<W: Write> Vp8Encoder<W> {
    fn new(writer: W) -> Self {
        let segment = Segment::default();

        Self {
            writer,
            frame: Frame::default(),
            encoder: ArithmeticEncoder::new(),
            segments: [segment; MAX_SEGMENTS],
            segments_enabled: false,

            loop_filter_adjustments: false,
            macroblock_no_skip_coeff: None,
            quantization_indices: QuantizationIndices::default(),

            token_probs: Default::default(),

            top_complexity: Vec::new(),
            left_complexity: Complexity::default(),

            top_b_pred: Vec::new(),
            left_b_pred: [IntraMode::default(); 4],

            macroblock_width: 0,
            macroblock_height: 0,

            partitions: vec![ArithmeticEncoder::new()],

            left_border_y: [0u8; 16 + 1],
            left_border_u: [0u8; 8 + 1],
            left_border_v: [0u8; 8 + 1],
            top_border_y: Vec::new(),
            top_border_u: Vec::new(),
            top_border_v: Vec::new(),
        }
    }

    /// Writes the uncompressed part of the frame header (9.1)
    fn write_uncompressed_frame_header(
        &mut self,
        partition_size: u32,
    ) -> Result<(), EncodingError> {
        let version = u32::from(self.frame.version);
        let for_display = if self.frame.for_display { 1 } else { 0 };

        let keyframe_bit = 0;
        let tag = (partition_size << 5) | (for_display << 4) | (version << 1) | (keyframe_bit);
        self.writer.write_u24::<LittleEndian>(tag)?;

        let magic_bytes_buffer: [u8; 3] = [0x9d, 0x01, 0x2a];
        self.writer.write_all(&magic_bytes_buffer)?;

        let width = self.frame.width & 0x3FFF;
        let height = self.frame.height & 0x3FFF;
        self.writer.write_u16::<LittleEndian>(width)?;
        self.writer.write_u16::<LittleEndian>(height)?;

        Ok(())
    }

    fn encode_compressed_frame_header(&mut self) {
        // if keyframe, color space must be 0
        self.encoder.write_literal(1, 0);
        // pixel type
        self.encoder.write_literal(1, 0);

        self.encoder.write_flag(self.segments_enabled);
        if self.segments_enabled {
            self.encode_segment_updates();
        }

        self.encoder.write_flag(self.frame.filter_type);
        self.encoder.write_literal(6, self.frame.filter_level);
        self.encoder.write_literal(3, self.frame.sharpness_level);

        self.encoder.write_flag(self.loop_filter_adjustments);
        if self.loop_filter_adjustments {
            self.encode_loop_filter_adjustments();
        }

        // partitions length must be 1, 2, 4 or 8, so value will be 0, 1, 2 or 3
        let partitions_value: u8 = self.partitions.len().ilog2().try_into().unwrap();
        self.encoder.write_literal(2, partitions_value);

        self.encode_quantization_indices();

        // refresh entropy probs
        self.encoder.write_literal(1, 0);

        self.encode_updated_token_probabilities();

        let mb_no_skip_coeff = if self.macroblock_no_skip_coeff.is_some() {
            1
        } else {
            0
        };
        self.encoder.write_literal(1, mb_no_skip_coeff);
        if let Some(prob_skip_false) = self.macroblock_no_skip_coeff {
            self.encoder.write_literal(8, prob_skip_false);
        }
    }

    fn write_partitions(&mut self) -> Result<(), EncodingError> {
        let partitions = std::mem::take(&mut self.partitions);
        let partitions_bytes: Vec<Vec<u8>> = partitions
            .into_iter()
            .map(|x| x.flush_and_get_buffer())
            .collect();
        // write the sizes of the partitions if there's more than 1
        if partitions_bytes.len() > 1 {
            for partition in partitions_bytes[..partitions_bytes.len() - 1].iter() {
                self.writer
                    .write_u24::<LittleEndian>(partition.len() as u32)?;
                self.writer.write_all(partition)?;
            }
        }

        // write the final partition
        self.writer
            .write_all(&partitions_bytes[partitions_bytes.len() - 1])?;

        Ok(())
    }

    fn encode_segment_updates(&mut self) {
        // TODO: encode this as per 9.3
        todo!();
    }

    fn encode_loop_filter_adjustments(&mut self) {
        // TODO: encode this
        todo!();
    }

    fn encode_quantization_indices(&mut self) {
        self.encoder
            .write_literal(7, self.quantization_indices.yac_abs);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.ydc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.y2dc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.y2ac_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.uvdc_delta);
        self.encoder
            .write_optional_signed_value(4, self.quantization_indices.uvac_delta);
    }

    // TODO: work out when we want to update these probabilities
    fn encode_updated_token_probabilities(&mut self) {
        for is in COEFF_UPDATE_PROBS.iter() {
            for js in is.iter() {
                for ks in js.iter() {
                    for prob in ks.iter() {
                        // currently just not updating these
                        self.encoder.write_bool(false, *prob);
                    }
                }
            }
        }
    }

    fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        if self.segments_enabled {
            if let Some(_segment_id) = macroblock_info.segment_id {
                // TODO: set segment for macroblock
                todo!();
            }
        }

        if let Some(prob) = self.macroblock_no_skip_coeff {
            self.encoder
                .write_bool(macroblock_info.coeffs_skipped, prob);
        }

        // encode macroblock info y mode using KEYFRAME_YMODE_TREE
        self.encoder.write_with_tree(
            &KEYFRAME_YMODE_TREE,
            &KEYFRAME_YMODE_PROBS,
            macroblock_info.luma_mode as i8,
        );

        match macroblock_info.luma_mode.into_intra() {
            None => {
                // 11.3 code each of the subblocks
                if let Some(bpred) = macroblock_info.luma_bpred {
                    for y in 0usize..4 {
                        let mut left = self.left_b_pred[y];
                        for x in 0usize..4 {
                            let top = self.top_b_pred[mbx * 4 + x];
                            let probs = &KEYFRAME_BPRED_MODE_PROBS[top as usize][left as usize];
                            let intra_mode = bpred[y * 4 + x];
                            self.encoder.write_with_tree(
                                &KEYFRAME_BPRED_MODE_TREE,
                                probs,
                                intra_mode as i8,
                            );
                            left = intra_mode;
                            self.top_b_pred[mbx * 4 + x] = intra_mode;
                        }
                        self.left_b_pred[y] = left;
                    }
                } else {
                    panic!("Invalid, can't set luma mode to B without setting preds");
                }
            }
            Some(intra_mode) => {
                for (left, top) in self
                    .left_b_pred
                    .iter_mut()
                    .zip(self.top_b_pred[4 * mbx..][..4].iter_mut())
                {
                    *left = intra_mode;
                    *top = intra_mode;
                }
            }
        }

        // encode macroblock info chroma mode
        self.encoder.write_with_tree(
            &KEYFRAME_UV_MODE_TREE,
            &KEYFRAME_UV_MODE_PROBS,
            macroblock_info.chroma_mode as i8,
        );
    }

    // 13 in specification, matches read_residual_data in the decoder
    fn encode_residual_data(
        &mut self,
        macroblock_info: &MacroblockInfo,
        partition_index: usize,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
    ) {
        let mut plane = if macroblock_info.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        // TODO: change to get index from macroblock
        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        // Y2
        if plane == Plane::Y2 {
            // encode 0th coefficient of each luma
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);

            // wht here on the 0th coeffs
            transform::wht4x4(&mut coeffs0);

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;

            let has_coeffs = self.encode_coefficients(
                &coeffs0,
                partition_index,
                plane,
                complexity.into(),
                segment.y2dc,
                segment.y2ac,
            );

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };

            // next encode luma coefficients without the 0th coeffs
            plane = Plane::YCoeff1;
        }

        // now encode the 16 luma 4x4 subblocks in the macroblock
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].y[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    segment.ydc,
                    segment.yac,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            // set for the next macroblock
            self.left_complexity.y[y] = left;
        }

        plane = Plane::Chroma;

        // encode the 4 u 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].u[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    segment.uvdc,
                    segment.uvac,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        // encode the 4 v 4x4 subblocks
        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].v[x];
                let complexity = left + top;

                let has_coeffs = self.encode_coefficients(
                    &block,
                    partition_index,
                    plane,
                    complexity.into(),
                    segment.uvdc,
                    segment.uvac,
                );

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }

    // encodes the coefficients which is the reverse procedure of read_coefficients in the decoder
    // returns whether there was any non-zero data in the block for the complexity
    fn encode_coefficients(
        &mut self,
        block: &[i32; 16],
        partition_index: usize,
        plane: Plane,
        complexity: usize,
        dc_quant: i16,
        ac_quant: i16,
    ) -> bool {
        // transform block
        // dc is used for the 0th coefficient, ac for the others

        let encoder = &mut self.partitions[partition_index];

        let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let probs = &self.token_probs[plane as usize];

        assert!(complexity <= 2);
        let mut complexity = complexity;

        // convert to zigzag and quantize
        // this is the only lossy part of the encoding
        let mut zigzag_block = [0i32; 16];
        for i in first_coeff..16 {
            let zigzag_index = usize::from(ZIGZAG[i]);
            let quant = if zigzag_index > 0 { ac_quant } else { dc_quant };
            zigzag_block[i] = block[zigzag_index] / i32::from(quant);
        }

        // get index of last coefficient that isn't 0
        let end_of_block_index =
            if let Some(last_non_zero_index) = zigzag_block.iter().rev().position(|x| *x != 0) {
                (15 - last_non_zero_index) + 1
            } else {
                // if it's all 0s then the first block is end of block
                0
            };

        let mut skip_eob = false;

        for index in first_coeff..end_of_block_index {
            let coeff = zigzag_block[index];

            let band = usize::from(COEFF_BANDS[index]);
            let probabilities = &probs[band][complexity];
            let start_index_token_tree = if skip_eob { 2 } else { 0 };
            let token_tree = &DCT_TOKEN_TREE;
            let token_probs = probabilities;

            let token = match coeff.abs() {
                0 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        DCT_0,
                        start_index_token_tree,
                    );

                    // never going to have an end of block after a 0, so skip checking next coeff
                    skip_eob = true;
                    DCT_0
                }

                // just encode as literal
                literal @ 1..=4 => {
                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        literal as i8,
                        start_index_token_tree,
                    );

                    skip_eob = false;
                    literal as i8
                }

                // encode the category
                value => {
                    let category = match value {
                        5..=6 => DCT_CAT1,
                        7..=10 => DCT_CAT2,
                        11..=18 => DCT_CAT3,
                        19..=34 => DCT_CAT4,
                        35..=66 => DCT_CAT5,
                        67..=2048 => DCT_CAT6,
                        _ => unreachable!(),
                    };

                    encoder.write_with_tree_start_index(
                        token_tree,
                        token_probs,
                        category,
                        start_index_token_tree,
                    );

                    let category_probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];

                    let extra = value - i32::from(DCT_CAT_BASE[(category - DCT_CAT1) as usize]);

                    let mut mask = if category == DCT_CAT6 {
                        1 << (11 - 1)
                    } else {
                        1 << (category - DCT_CAT1)
                    };

                    for &prob in category_probs.iter() {
                        if prob == 0 {
                            break;
                        }
                        let extra_bool = extra & mask > 0;
                        encoder.write_bool(extra_bool, prob);
                        mask >>= 1;
                    }

                    skip_eob = false;

                    category
                }
            };

            // encode sign if token is not zero
            if token != DCT_0 {
                // note flag means coeff is negative
                encoder.write_flag(!coeff.is_positive());
            }

            complexity = match token {
                DCT_0 => 0,
                DCT_1 => 1,
                _ => 2,
            };
        }

        // encode end of block
        if end_of_block_index < 16 {
            let band_index = usize::max(first_coeff, end_of_block_index);
            let band = usize::from(COEFF_BANDS[band_index]);
            let probabilities = &probs[band][complexity];
            encoder.write_with_tree(&DCT_TOKEN_TREE, probabilities, DCT_EOB);
        }

        // whether the block has a non zero coefficient
        end_of_block_index > 0
    }

    fn encode_image(
        &mut self,
        data: &[u8],
        color: ColorType,
        width: u16,
        height: u16,
        lossy_quality: u8,
    ) -> Result<(), EncodingError> {
        let (y_bytes, u_bytes, v_bytes) = match color {
            ColorType::Rgb8 => convert_image_yuv::<3>(data, width, height),
            ColorType::Rgba8 => convert_image_yuv::<4>(data, width, height),
            ColorType::L8 => convert_image_y::<1>(data, width, height),
            ColorType::La8 => convert_image_y::<2>(data, width, height),
        };

        let bytes_per_pixel = match color {
            ColorType::L8 => 1,
            ColorType::La8 => 2,
            ColorType::Rgb8 => 3,
            ColorType::Rgba8 => 4,
        };
        assert_eq!(
            (u64::from(width) * u64::from(height)).saturating_mul(bytes_per_pixel),
            data.len() as u64,
            "width/height doesn't match data length of {} for the color type {:?}",
            data.len(),
            color
        );

        self.setup_encoding(lossy_quality, width, height, y_bytes, u_bytes, v_bytes);

        // Learn the real skip rate before writing the frame header, which is
        // where `prob_skip_false` has to go (9.10/19.2) - it comes before any
        // macroblock data in the bitstream. See `count_skipped_macroblocks`
        // and `skip_probability` for why this dry run is both correct and
        // worth its cost.
        let (skipped, total) = self.count_skipped_macroblocks();
        self.reset_frame_state();
        self.macroblock_no_skip_coeff = Some(skip_probability(skipped, total));

        self.encode_compressed_frame_header();

        // encode residual partitions first
        for mby in 0..self.macroblock_height {
            let partition_index = usize::from(mby) % self.partitions.len();
            // reset left complexity / bpreds for left of image
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];

            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let macroblock_info = self.choose_macroblock_info(mbx.into(), mby.into());

                // write macroblock headers
                self.write_macroblock_header(&macroblock_info, mbx.into());

                // Reconstruction always has to run, skipped or not: even a
                // skipped macroblock's borders (what the *next* macroblock
                // predicts from) are the prediction with a zero residual
                // added, which is exactly what `transform_luma_block` /
                // `transform_chroma_blocks` compute - the decoder does the
                // same thing unconditionally (`decode_frame_` in mod.rs always
                // calls `intra_predict_luma`/`intra_predict_chroma`, just with
                // an all-zero coefficient block when `coeffs_skipped`). Only
                // the bit-writing below is conditional.
                let y_block_data =
                    self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                let (u_block_data, v_block_data) = self.transform_chroma_blocks(
                    mbx.into(),
                    mby.into(),
                    macroblock_info.chroma_mode,
                );

                if !macroblock_info.coeffs_skipped {
                    self.encode_residual_data(
                        &macroblock_info,
                        partition_index,
                        mbx as usize,
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                } else {
                    // since coeffs are all zero, need to set all complexities to 0
                    // except if the luma mode is B then won't set Y2
                    self.left_complexity
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                    self.top_complexity[usize::from(mbx)]
                        .clear(macroblock_info.luma_mode != LumaMode::B);
                }
            }
        }

        let compressed_header_encoder = std::mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32)?;

        self.writer.write_all(&compressed_header_bytes)?;

        self.write_partitions()?;

        Ok(())
    }

    /// The four "whole macroblock" luma prediction modes considered by RD
    /// mode decision. `LumaMode::B` (independent-per-sub-block prediction)
    /// is deliberately not in this list - it needs its own per-block search
    /// with sequential reconstruction and `left_b_pred`/`top_b_pred` context
    /// tracking, which `trial_luma_bpred` does separately; its result is
    /// compared against the winner of this list in `choose_macroblock_info`.
    const LUMA_MODE_CANDIDATES: [LumaMode; 4] =
        [LumaMode::DC, LumaMode::V, LumaMode::H, LumaMode::TM];

    const CHROMA_MODE_CANDIDATES: [ChromaMode; 4] =
        [ChromaMode::DC, ChromaMode::V, ChromaMode::H, ChromaMode::TM];

    /// The ten 4x4 intra prediction modes considered for each B_PRED
    /// sub-block by `trial_luma_bpred`. A distinct type (`IntraMode`) and a
    /// distinct search from `LUMA_MODE_CANDIDATES` above: one `IntraMode` is
    /// chosen per 4x4 sub-block (16 per macroblock) rather than once for the
    /// whole macroblock.
    const BPRED_MODE_CANDIDATES: [IntraMode; 10] = [
        IntraMode::DC,
        IntraMode::TM,
        IntraMode::VE,
        IntraMode::HE,
        IntraMode::LD,
        IntraMode::RD,
        IntraMode::VR,
        IntraMode::VL,
        IntraMode::HD,
        IntraMode::HU,
    ];

    /// Picks the luma and chroma prediction modes for one macroblock by
    /// Lagrangian RD cost (`distortion + lambda * rate`, see
    /// `mode_decision_lambda`), and whether the macroblock can be signalled
    /// as skipped (all quantized coefficients zero).
    ///
    /// This method is deliberately `&self`, not `&mut self`: every trial
    /// below runs predict -> residual -> DCT -> quantize -> dequantize ->
    /// IDCT -> reconstruct against *local* copies of the prediction buffers,
    /// and reads `top_border_*` / `left_border_*` / `self.frame` without
    /// touching them. The real, state-mutating version of this pipeline
    /// (`transform_luma_block` / `transform_chroma_blocks`) runs exactly
    /// once per macroblock, in `encode_image`, using the mode this function
    /// returns - so the borders the *next* macroblock predicts from are
    /// always the actual reconstruction of the chosen mode, never a trial.
    fn choose_macroblock_info(&self, mbx: usize, mby: usize) -> MacroblockInfo {
        let segment = self.segments[0];
        let lambda = mode_decision_lambda(&segment);

        let mut best_luma: Option<(f64, LumaMode, Luma16x16Coeffs)> = None;
        for &mode in &Self::LUMA_MODE_CANDIDATES {
            let (distortion, rate, coeffs) = self.trial_luma_16x16(mode, mbx, mby, &segment);
            let cost = distortion as f64 + lambda * rate as f64;
            let better = match &best_luma {
                None => true,
                Some((best_cost, ..)) => cost < *best_cost,
            };
            if better {
                best_luma = Some((cost, mode, coeffs));
            }
        }
        let (best_16x16_cost, luma_mode_16x16, luma_coeffs_16x16) =
            best_luma.expect("LUMA_MODE_CANDIDATES is non-empty");

        // B_PRED candidate: a sequential, per-sub-block search (see
        // `trial_luma_bpred`'s doc comment for why it can't be scored the
        // same way as the four whole-macroblock modes above), compared
        // against the best of those four by the exact same Lagrangian cost
        // so the comparison is apples to apples.
        let (bpred_distortion, bpred_rate, bpred_modes, bpred_coeffs) =
            self.trial_luma_bpred(mbx, mby, &segment, lambda);
        let bpred_cost = bpred_distortion as f64 + lambda * bpred_rate;

        // B_PRED has no Y2 block (`encode_residual_data` branches on this
        // exact condition), so its "all zero" check only looks at the 16
        // sub-blocks' own coefficients - there is no separate Y2 plane to
        // also check, unlike the 16x16 modes just below.
        let (luma_mode, luma_bpred, luma_all_zero) = if bpred_cost < best_16x16_cost {
            (
                LumaMode::B,
                Some(bpred_modes),
                bpred_coeffs.iter().all(|&c| c == 0),
            )
        } else {
            (
                luma_mode_16x16,
                None,
                luma_coeffs_16x16.y2_coeffs.iter().all(|&c| c == 0)
                    && luma_coeffs_16x16.y_coeffs.iter().all(|&c| c == 0),
            )
        };

        let mut best_chroma: Option<(f64, ChromaMode, ChromaCoeffs, ChromaCoeffs)> = None;
        for &mode in &Self::CHROMA_MODE_CANDIDATES {
            let (distortion, rate, u_coeffs, v_coeffs) = self.trial_chroma(mode, mbx, mby);
            let cost = distortion as f64 + lambda * rate as f64;
            let better = match &best_chroma {
                None => true,
                Some((best_cost, ..)) => cost < *best_cost,
            };
            if better {
                best_chroma = Some((cost, mode, u_coeffs, v_coeffs));
            }
        }
        let (_, chroma_mode, u_coeffs, v_coeffs) =
            best_chroma.expect("CHROMA_MODE_CANDIDATES is non-empty");

        // The macroblock can be signalled as skipped iff the chosen modes
        // leave every quantized coefficient at zero - luma (`luma_all_zero`,
        // computed above from either the 16x16 y2+y coefficients or the
        // B_PRED sub-blocks' own coefficients, whichever mode won) and both
        // chroma planes. `write_macroblock_header` / `encode_image` are what
        // actually act on this flag; this is purely "would every block
        // we're about to write be empty".
        let coeffs_skipped =
            luma_all_zero && u_coeffs.iter().all(|&c| c == 0) && v_coeffs.iter().all(|&c| c == 0);

        MacroblockInfo {
            luma_mode,
            luma_bpred,
            chroma_mode,
            segment_id: None,
            coeffs_skipped,
        }
    }

    /// RD trial for one candidate 16x16 luma prediction mode: predicts,
    /// takes the residual against the source, DCTs, quantizes, and then
    /// dequantizes/IDCTs/reconstructs *into a local buffer* to measure how
    /// close the reconstruction gets (distortion) and how expensive its
    /// coefficients are (rate). Never mutates `self`.
    fn trial_luma_16x16(
        &self,
        luma_mode: LumaMode,
        mbx: usize,
        mby: usize,
        segment: &Segment,
    ) -> (i64, i64, Luma16x16Coeffs) {
        let y_with_border = self.get_predicted_luma_block_16x16(luma_mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&y_with_border, mbx, mby);
        let mut coeffs = self.get_luma_block_coeffs_16x16(luma_blocks, segment);
        let dequantized_blocks = self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs);

        // Reconstruct into a copy of the predicted block so this trial never
        // touches `self.top_border_y` / `self.left_border_y`.
        let mut recon = y_with_border;
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = x + y * 4;
                let rb: &[i32; 16] = dequantized_blocks[i * 16..][..16].try_into().unwrap();
                add_residue(&mut recon, rb, 1 + y * 4, 1 + x * 4, LUMA_STRIDE);
            }
        }

        let distortion = self.luma_sse(&recon, mbx, mby);
        let rate = coeff_rate_estimate(&coeffs.y2_coeffs) + coeff_rate_estimate(&coeffs.y_coeffs);

        (distortion, rate, coeffs)
    }

    /// Sum of squared error between a trial reconstruction of a 16x16 luma
    /// macroblock (as produced by `trial_luma_16x16`, still carrying its
    /// 1-pixel border) and the true source pixels.
    fn luma_sse(&self, recon: &[u8; LUMA_BLOCK_SIZE], mbx: usize, mby: usize) -> i64 {
        let stride = LUMA_STRIDE;
        let width = usize::from(self.macroblock_width) * 16;
        let mut sse: i64 = 0;
        for y in 0..16 {
            for x in 0..16 {
                let r = i64::from(recon[(y + 1) * stride + (x + 1)]);
                let a = i64::from(self.frame.ybuf[(mby * 16 + y) * width + mbx * 16 + x]);
                let d = r - a;
                sse += d * d;
            }
        }
        sse
    }

    /// RD trial for B_PRED (4x4 sub-block) luma prediction.
    ///
    /// Unlike the four 16x16 candidates `trial_luma_16x16` scores
    /// independently, the 16 sub-blocks here are *not* independent: VP8's
    /// 4x4 intra predictors (`predict_b*pred` in `prediction.rs`) read the
    /// reconstructed pixels immediately above and to the left of each
    /// sub-block, and for sub-blocks inside this macroblock those
    /// neighbours are other sub-blocks of this same macroblock. So this
    /// walks the 16 sub-blocks in raster order - top-to-bottom,
    /// left-to-right, the same order `transform_luma_blocks_4x4` and the
    /// decoder use - and after picking the cheapest mode for a sub-block it
    /// commits that sub-block's quantized/dequantized reconstruction into a
    /// local border buffer *before* evaluating the next sub-block. Scoring
    /// all 16 independently against the original source pixels would let
    /// each sub-block predict from source pixels the decoder never has
    /// (only the lossy reconstruction of its neighbours), which would
    /// silently and permanently desync every macroblock that predicts from
    /// this one onward - exactly the drift the module-level design notes
    /// warn about.
    ///
    /// Like `trial_luma_16x16`, this never mutates `self`: the border
    /// buffer it reconstructs into is local (seeded from, but never written
    /// back to, `self.top_border_y` / `self.left_border_y`), and the
    /// B_PRED submode context it reads (`self.top_b_pred` /
    /// `self.left_b_pred`) is copied into local variables before the
    /// per-sub-block loop starts, rather than updated in place.
    ///
    /// Returns:
    /// - the summed distortion (SSE, pixel domain, same units
    ///   `trial_luma_16x16` returns),
    /// - the summed rate: coefficient rate (`coeff_rate_estimate`, the same
    ///   proxy `trial_luma_16x16` uses) plus submode signalling cost
    ///   (`bpred_mode_bit_cost`, real entropy bits - see that function's
    ///   doc comment for why these two are on different scales and what
    ///   that means for how `lambda` weighs them),
    /// - the 16 chosen submodes in raster order (for
    ///   `MacroblockInfo::luma_bpred` / `write_macroblock_header`), and
    /// - their quantized coefficients (to test whether the macroblock can
    ///   be signalled skipped - B_PRED has no Y2 block, so unlike
    ///   `trial_luma_16x16`'s `Luma16x16Coeffs` there is nothing else to
    ///   check).
    fn trial_luma_bpred(
        &self,
        mbx: usize,
        mby: usize,
        segment: &Segment,
        lambda: f64,
    ) -> (i64, f64, [IntraMode; 16], LumaYCoeffs) {
        let stride = LUMA_STRIDE;
        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        // Running B_PRED context, local copies of `self.top_b_pred` /
        // `self.left_b_pred` updated as sub-blocks are chosen - mirrors
        // exactly what `write_macroblock_header` does when it later writes
        // this macroblock for real (see its `left`/`top` bookkeeping), but
        // `top_ctx` is the only one that needs to persist and mutate across
        // the loop: the "left" context for each row starts fresh from
        // `self.left_b_pred[row]` (the neighbouring macroblock to the
        // left), same as `write_macroblock_header`.
        let mut top_ctx: [IntraMode; 4] = self.top_b_pred[mbx * 4..][..4].try_into().unwrap();

        let mut total_distortion: i64 = 0;
        let mut total_rate: f64 = 0.0;
        let mut chosen_modes = [IntraMode::default(); 16];
        let mut y_coeffs: LumaYCoeffs = [0i32; 16 * 16];

        for sby in 0usize..4 {
            let mut left = self.left_b_pred[sby];
            // `sbx` indexes `top_ctx` but is also used directly to derive
            // `x0`/`i` below, so an iterator/enumerate rewrite would need
            // its own counter anyway.
            #[allow(clippy::needless_range_loop)]
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;
                let top = top_ctx[sbx];
                let y_data_block_index = (mby * 16 + sby * 4) * width + mbx * 16 + sbx * 4;

                // (cost, mode, distortion, rate, quantized coeffs, the raw
                // predicted 4x4 block, the dequantized residual) of the
                // best candidate seen so far for this sub-block.
                #[allow(clippy::type_complexity)]
                let mut best: Option<(f64, IntraMode, i64, f64, [i32; 16], [u8; 16], [i32; 16])> =
                    None;

                for &mode in &Self::BPRED_MODE_CANDIDATES {
                    // Every 4x4 predictor reads only border pixels that are
                    // already committed - either from outside this
                    // macroblock, or from an earlier sub-block in raster
                    // order - and writes only its own 4x4 interior. So
                    // trying candidates back to back on the same buffer is
                    // safe: nothing one candidate's prediction writes is
                    // ever read by the next candidate's prediction.
                    match mode {
                        IntraMode::TM => predict_tmpred(&mut y_with_border, 4, x0, y0, stride),
                        IntraMode::VE => predict_bvepred(&mut y_with_border, x0, y0, stride),
                        IntraMode::HE => predict_bhepred(&mut y_with_border, x0, y0, stride),
                        IntraMode::DC => predict_bdcpred(&mut y_with_border, x0, y0, stride),
                        IntraMode::LD => predict_bldpred(&mut y_with_border, x0, y0, stride),
                        IntraMode::RD => predict_brdpred(&mut y_with_border, x0, y0, stride),
                        IntraMode::VR => predict_bvrpred(&mut y_with_border, x0, y0, stride),
                        IntraMode::VL => predict_bvlpred(&mut y_with_border, x0, y0, stride),
                        IntraMode::HD => predict_bhdpred(&mut y_with_border, x0, y0, stride),
                        IntraMode::HU => predict_bhupred(&mut y_with_border, x0, y0, stride),
                    }

                    let mut predicted_block = [0u8; 16];
                    let mut residual = [0i32; 16];
                    for y in 0..4 {
                        for x in 0..4 {
                            let border_index = (y0 + y) * stride + x0 + x;
                            let predicted_value = y_with_border[border_index];
                            let actual_value =
                                self.frame.ybuf[y_data_block_index + y * width + x];
                            predicted_block[y * 4 + x] = predicted_value;
                            residual[y * 4 + x] =
                                i32::from(actual_value) - i32::from(predicted_value);
                        }
                    }

                    transform::dct4x4(&mut residual);

                    // quantize
                    let mut quantized = residual;
                    for (index, v) in quantized.iter_mut().enumerate() {
                        let quant = if index > 0 { segment.yac } else { segment.ydc };
                        *v /= i32::from(quant);
                    }

                    // Dequantize and inverse-transform, matching the
                    // round-trip `transform_luma_blocks_4x4` performs
                    // exactly, so this trial's reconstruction is the one
                    // the real pass (and the decoder) will actually
                    // produce.
                    let mut dequantized = quantized;
                    for (index, v) in dequantized.iter_mut().enumerate() {
                        let quant = if index > 0 { segment.yac } else { segment.ydc };
                        *v *= i32::from(quant);
                    }
                    transform::idct4x4(&mut dequantized);

                    let mut distortion: i64 = 0;
                    for y in 0..4 {
                        for x in 0..4 {
                            let p = i32::from(predicted_block[y * 4 + x]);
                            let r = (p + dequantized[y * 4 + x]).clamp(0, 255);
                            let actual = i64::from(
                                self.frame.ybuf[y_data_block_index + y * width + x],
                            );
                            let d = i64::from(r) - actual;
                            distortion += d * d;
                        }
                    }

                    let coeff_rate = coeff_rate_estimate(&quantized) as f64;
                    let submode_rate = bpred_mode_bit_cost(top, left, mode);
                    let rate = coeff_rate + submode_rate;
                    let cost = distortion as f64 + lambda * rate;

                    let better = match &best {
                        None => true,
                        Some((best_cost, ..)) => cost < *best_cost,
                    };
                    if better {
                        best = Some((
                            cost,
                            mode,
                            distortion,
                            rate,
                            quantized,
                            predicted_block,
                            dequantized,
                        ));
                    }
                }

                let (_, mode, distortion, rate, quantized, predicted_block, dequantized) =
                    best.expect("BPRED_MODE_CANDIDATES is non-empty");

                // Commit the winning sub-block's reconstruction into the
                // shared border buffer before moving on to the next
                // sub-block - this is exactly what makes later sub-blocks
                // (and, once this mode is chosen for real, later
                // macroblocks) predict from the same lossy reconstruction
                // the decoder will have, rather than from source pixels.
                for y in 0..4 {
                    for x in 0..4 {
                        let border_index = (y0 + y) * stride + x0 + x;
                        let p = i32::from(predicted_block[y * 4 + x]);
                        y_with_border[border_index] =
                            (p + dequantized[y * 4 + x]).clamp(0, 255) as u8;
                    }
                }

                chosen_modes[i] = mode;
                y_coeffs[i * 16..][..16].copy_from_slice(&quantized);
                total_distortion += distortion;
                total_rate += rate;

                left = mode;
                top_ctx[sbx] = mode;
            }
        }

        (total_distortion, total_rate, chosen_modes, y_coeffs)
    }

    /// RD trial for one candidate chroma prediction mode, covering both U
    /// and V (a macroblock has a single `ChromaMode` shared by both planes).
    /// Same shape as `trial_luma_16x16`: never mutates `self`.
    fn trial_chroma(
        &self,
        chroma_mode: ChromaMode,
        mbx: usize,
        mby: usize,
    ) -> (i64, i64, ChromaCoeffs, ChromaCoeffs) {
        let mut predicted_u = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_u,
            &self.left_border_u,
        );
        let mut predicted_v = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_v,
            &self.left_border_v,
        );

        let u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

        let u_coeffs = self.get_chroma_block_coeffs(u_blocks);
        let v_coeffs = self.get_chroma_block_coeffs(v_blocks);

        let dequantized_u = self.get_dequantized_blocks_from_coeffs_chroma(&u_coeffs);
        let dequantized_v = self.get_dequantized_blocks_from_coeffs_chroma(&v_coeffs);

        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = dequantized_u[i * 16..][..16].try_into().unwrap();
                add_residue(&mut predicted_u, urb, 1 + y * 4, 1 + x * 4, CHROMA_STRIDE);

                let vrb: &[i32; 16] = dequantized_v[i * 16..][..16].try_into().unwrap();
                add_residue(&mut predicted_v, vrb, 1 + y * 4, 1 + x * 4, CHROMA_STRIDE);
            }
        }

        let distortion = self.chroma_sse(&predicted_u, &self.frame.ubuf, mbx, mby)
            + self.chroma_sse(&predicted_v, &self.frame.vbuf, mbx, mby);
        let rate = coeff_rate_estimate(&u_coeffs) + coeff_rate_estimate(&v_coeffs);

        (distortion, rate, u_coeffs, v_coeffs)
    }

    /// Sum of squared error between a trial reconstruction of one 8x8 chroma
    /// plane (still carrying its 1-pixel border) and the true source pixels
    /// of that plane (`self.frame.ubuf` or `self.frame.vbuf`).
    fn chroma_sse(&self, recon: &[u8; CHROMA_BLOCK_SIZE], plane: &[u8], mbx: usize, mby: usize) -> i64 {
        let stride = CHROMA_STRIDE;
        let chroma_width = usize::from(self.macroblock_width) * 8;
        let mut sse: i64 = 0;
        for y in 0..8 {
            for x in 0..8 {
                let r = i64::from(recon[(y + 1) * stride + (x + 1)]);
                let a = i64::from(plane[(mby * 8 + y) * chroma_width + mbx * 8 + x]);
                let d = r - a;
                sse += d * d;
            }
        }
        sse
    }

    /// Dry run of the whole frame's macroblock loop - mode decision plus the
    /// real, state-mutating reconstruction - counting what fraction of
    /// macroblocks end up skippable, without writing any header or residual
    /// bits. Used only to pick `prob_skip_false` (see `skip_probability`)
    /// before the frame header is written, since that header comes before
    /// any macroblock data in the bitstream and so has to be decided upfront.
    ///
    /// This intentionally reuses `choose_macroblock_info` /
    /// `transform_luma_block` / `transform_chroma_blocks` verbatim rather
    /// than re-implementing a cheaper estimate: mode decision only depends
    /// on pixel data and border state, never on entropy-coding
    /// probabilities, so the skip decisions made here are exactly the ones
    /// the real pass will make. The border/b_pred/complexity state this
    /// mutates is fully reset by `reset_frame_state` immediately afterwards.
    fn count_skipped_macroblocks(&mut self) -> (u32, u32) {
        let mut total = 0u32;
        let mut skipped = 0u32;

        for mby in 0..self.macroblock_height {
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];
            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let info = self.choose_macroblock_info(mbx.into(), mby.into());
                self.transform_luma_block(mbx.into(), mby.into(), &info);
                self.transform_chroma_blocks(mbx.into(), mby.into(), info.chroma_mode);

                total += 1;
                if info.coeffs_skipped {
                    skipped += 1;
                }
            }
        }

        (skipped, total)
    }

    // sets up the encoding of the encoder by setting all the encoder params based on the width and height
    fn setup_encoding(
        &mut self,
        lossy_quality: u8,
        width: u16,
        height: u16,
        y_buf: Vec<u8>,
        u_buf: Vec<u8>,
        v_buf: Vec<u8>,
    ) {
        let mb_width = width.div_ceil(16);
        let mb_height = height.div_ceil(16);
        self.macroblock_width = mb_width;
        self.macroblock_height = mb_height;
        self.frame = Frame {
            width,
            height,

            ybuf: y_buf,
            ubuf: u_buf,
            vbuf: v_buf,

            version: 0,

            keyframe: true,
            for_display: true,
            pixel_type: 0,

            filter_type: false,
            // MUST stay 0 until the encoder applies the loop filter to its
            // own reconstruction.
            //
            // VP8's loop filter is *in-loop*: the decoder filters each
            // reconstructed macroblock before later macroblocks predict from
            // it. This encoder never does - `loop_filter::` is called only
            // from the decoder, in `lossy/mod.rs` - so it predicts from
            // unfiltered pixels while the decoder predicts from filtered
            // ones. Signalling a non-zero level guarantees encoder/decoder
            // drift, and the drift grows with the level.
            //
            // This was hardcoded to 63, the maximum, which maximised it.
            // Measured with examples/rd_eval.rs on six Kodak images at three
            // DSSIM targets, size relative to libwebp (lower is better):
            //
            //     image      <=0.0150        <=0.0080        <=0.0035
            //     kodim01  1.51 -> 1.38    1.64 -> 1.58    1.97 -> 1.97
            //     kodim02  1.57 -> 1.57    1.51 -> 1.45    2.03 -> 1.71
            //     kodim03  1.89 -> 1.89    1.92 -> 1.92    2.19 -> 2.19
            //     kodim05  2.25 -> 2.09    2.41 -> 2.25    2.65 -> 2.53
            //     kodim13  1.99 -> 1.86    2.09 -> 2.09    2.33 -> 2.20
            //     kodim19  1.74 -> 1.69    1.79 -> 1.79    2.10 -> 1.97
            //
            // Mean -4.0%, median -3.8%, best -15.8%, and worse on 0 of 18
            // points. A monotonic sweep (0, 8, 16, 32, 63) confirms the cost
            // rises with the level, which is the drift signature rather than
            // a quality trade.
            //
            // The real fix is to filter the reconstruction here and then
            // derive a level from the quantiser the way libwebp does; that
            // should beat 0, because the filter exists to help prediction.
            // Until then 0 is the only value that is not actively wrong.
            filter_level: 0,
            sharpness_level: 7,
        };

        self.token_probs = COEFF_PROBS;

        // choosing the quantization quality based on the quality passed in
        if lossy_quality > 100 {
            panic!("lossy quality must be between 0 and 100");
        }

        let quant_index: u8 = (127 - u16::from(lossy_quality) * 127 / 100) as u8;
        let quant_index_usize: usize = quant_index as usize;

        self.segments_enabled = false;
        let quantization_indices = QuantizationIndices {
            yac_abs: quant_index,
            ..Default::default()
        };
        self.quantization_indices = quantization_indices;

        let segment = Segment {
            ydc: DC_QUANT[quant_index_usize],
            yac: AC_QUANT[quant_index_usize],
            y2dc: DC_QUANT[quant_index_usize] * 2,
            y2ac: ((i32::from(AC_QUANT[quant_index_usize]) * 155 / 100) as i16).max(8),
            uvdc: DC_QUANT[quant_index_usize],
            uvac: AC_QUANT[quant_index_usize],
            ..Default::default()
        };
        self.segments[0] = segment;

        self.reset_frame_state();
    }

    /// Resets every piece of per-frame encoding state that prediction reads
    /// (borders, B_PRED context, coefficient complexity) back to its
    /// initial, "no macroblocks encoded yet" values. Called both from
    /// `setup_encoding` (first use) and from `encode_image` after the dry
    /// `count_skipped_macroblocks` pass (which mutates all of this exactly
    /// like the real pass would, and must not leak into it).
    fn reset_frame_state(&mut self) {
        let mb_width = self.macroblock_width;

        self.top_complexity = vec![Complexity::default(); usize::from(mb_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        self.left_b_pred = [IntraMode::default(); 4];

        self.left_border_y = [129u8; 16 + 1];
        self.left_border_u = [129u8; 8 + 1];
        self.left_border_v = [129u8; 8 + 1];

        self.top_border_y = vec![127u8; usize::from(mb_width) * 16 + 4];
        self.top_border_u = vec![127u8; usize::from(mb_width) * 8];
        self.top_border_v = vec![127u8; usize::from(mb_width) * 8];
    }

    // this is for all the luma modes except B
    fn get_predicted_luma_block_16x16(
        &self,
        luma_mode: LumaMode,
        mbx: usize,
        mby: usize,
    ) -> [u8; LUMA_BLOCK_SIZE] {
        let stride = LUMA_STRIDE;

        let mbw = self.macroblock_width;

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        // do the prediction
        match luma_mode {
            LumaMode::V => predict_vpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::H => predict_hpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::TM => predict_tmpred(&mut y_with_border, 16, 1, 1, stride),
            LumaMode::DC => predict_dcpred(&mut y_with_border, 16, stride, mby != 0, mbx != 0),
            LumaMode::B => unreachable!(),
        }

        y_with_border
    }

    // gets the luma blocks with the DCT applied to them
    fn get_luma_blocks_from_predicted_16x16(
        &self,
        predicted_y_block: &[u8; LUMA_BLOCK_SIZE],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let stride = LUMA_STRIDE;
        let width = usize::from(self.macroblock_width * 16);
        let mut luma_blocks = [0i32; 16 * 16];

        for block_y in 0..4 {
            for block_x in 0..4 {
                // the index on the luma block
                let block_index = block_y * 16 * 4 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let y_data_block_index = (mby * 16 + block_y * 4) * width + mbx * 16 + block_x * 4;

                let mut block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let predicted_value = predicted_y_block[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = self.frame.ybuf[actual_index];
                        block[y * 4 + x] = i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                // transform block before copying it into main block
                transform::dct4x4(&mut block);

                luma_blocks[block_index..][..16].copy_from_slice(&block);
            }
        }

        luma_blocks
    }

    // converts the predicted y block to the coeffs
    fn get_luma_block_coeffs_16x16(
        &self,
        mut luma_blocks: [i32; 16 * 16],
        segment: &Segment,
    ) -> Luma16x16Coeffs {
        let mut coeffs0 = get_coeffs0_from_block(&luma_blocks);
        // wht transform the y2 block and quantize it
        transform::wht4x4(&mut coeffs0);
        for (index, value) in coeffs0.iter_mut().enumerate() {
            let quant = if index > 0 {
                segment.y2ac
            } else {
                segment.y2dc
            };
            *value /= i32::from(quant);
        }

        // quantize the y blocks
        for y_block in luma_blocks.chunks_exact_mut(16) {
            for (index, y_value) in y_block.iter_mut().enumerate() {
                if index == 0 {
                    *y_value = 0;
                } else {
                    *y_value /= i32::from(segment.yac);
                }
            }
        }

        Luma16x16Coeffs {
            y2_coeffs: coeffs0,
            y_coeffs: luma_blocks,
        }
    }

    fn get_dequantized_blocks_from_coeffs_luma_16x16(
        &self,
        coeffs: &mut Luma16x16Coeffs,
    ) -> [i32; 16 * 16] {
        let mut dequantized_luma_residue = [0i32; 16 * 16];
        let segment = self.segments[0];

        for (k, y2_coeff) in coeffs.y2_coeffs.iter_mut().enumerate() {
            let quant = if k > 0 { segment.y2ac } else { segment.y2dc };
            *y2_coeff *= i32::from(quant);
        }
        transform::iwht4x4(&mut coeffs.y2_coeffs);

        // de-quantize the y blocks as well as do the inverse transform
        for (k, luma_block) in coeffs.y_coeffs.chunks_exact_mut(16).enumerate() {
            for y_value in luma_block[1..].iter_mut() {
                *y_value *= i32::from(segment.yac);
            }

            luma_block[0] = coeffs.y2_coeffs[k];

            transform::idct4x4(luma_block);

            dequantized_luma_residue[k * 16..][..16].copy_from_slice(luma_block);
        }

        dequantized_luma_residue
    }

    // Transforms the luma macroblock in the following ways
    // 1. Does the luma prediction and subtracts from the block
    // 2. Converts the block so each 4x4 subblock is contiguous within the block
    // 3. Does the DCT on each subblock
    // 4. Quantizes the block and dequantizes each subblock
    // 5. Calculates the quantized block - this can be used to calculate how accurate the
    // result is and is used to populate the borders for the next macroblock
    fn transform_luma_block(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> [i32; 16 * 16] {
        if macroblock_info.luma_mode == LumaMode::B {
            if let Some(bpred_modes) = macroblock_info.luma_bpred {
                return self.transform_luma_blocks_4x4(bpred_modes, mbx, mby);
            } else {
                panic!("Invalid, need bpred modes for luma mode B");
            }
        }

        let mut y_with_border =
            self.get_predicted_luma_block_16x16(macroblock_info.luma_mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&y_with_border, mbx, mby);

        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        // get coeffs
        let mut coeffs = self.get_luma_block_coeffs_16x16(luma_blocks, &segment);

        // now we're essentially applying the same functions as the decoder in order to ensure
        // that the border is the same as the one used for the decoder in the same macroblock
        let dequantized_blocks = self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs);

        // re-use the y_with_border from earlier since the prediction is still valid
        // applies the same thing as the decoder so that the border will line up
        for y in 0usize..4 {
            for x in 0usize..4 {
                let i = x + y * 4;
                // Create a reference to a [i32; 16] array for add_residue (slices of size 16 do not work).
                let rb: &[i32; 16] = dequantized_blocks[i * 16..][..16].try_into().unwrap();
                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;

                add_residue(&mut y_with_border, rb, y0, x0, LUMA_STRIDE);
            }
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * LUMA_STRIDE + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * LUMA_STRIDE + x + 1];
        }

        luma_blocks
    }

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 16] {
        let mut luma_blocks = [0i32; 16 * 16];
        let stride = 1usize + 16 + 4;
        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let mut y_with_border = create_border_luma(
            mbx,
            mby,
            mbw.into(),
            &self.top_border_y,
            &self.left_border_y,
        );

        let segment = self.segments[0];

        for sby in 0usize..4 {
            for sbx in 0usize..4 {
                let i = sby * 4 + sbx;
                let y0 = sby * 4 + 1;
                let x0 = sbx * 4 + 1;

                match bpred_modes[i] {
                    IntraMode::TM => predict_tmpred(&mut y_with_border, 4, x0, y0, stride),
                    IntraMode::VE => predict_bvepred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HE => predict_bhepred(&mut y_with_border, x0, y0, stride),
                    IntraMode::DC => predict_bdcpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::LD => predict_bldpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::RD => predict_brdpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::VR => predict_bvrpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::VL => predict_bvlpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HD => predict_bhdpred(&mut y_with_border, x0, y0, stride),
                    IntraMode::HU => predict_bhupred(&mut y_with_border, x0, y0, stride),
                }

                let block_index = sby * 16 * 4 + sbx * 16;
                let mut current_subblock = [0i32; 16];

                // subtract actual values here
                let border_subblock_index = y0 * stride + x0;
                let y_data_block_index = (mby * 16 + sby * 4) * width + mbx * 16 + sbx * 4;
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_subblock_index + y * stride + x;
                        let predicted_value = y_with_border[predicted_index];
                        let actual_index = y_data_block_index + y * width + x;
                        let actual_value = self.frame.ybuf[actual_index];
                        current_subblock[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut current_subblock);

                luma_blocks[block_index..][..16].copy_from_slice(&current_subblock);

                // quantize and de-quantize the subblock
                for (index, y_value) in current_subblock.iter_mut().enumerate() {
                    let quant = if index > 0 { segment.yac } else { segment.ydc };
                    *y_value = (*y_value / i32::from(quant)) * i32::from(quant);
                }
                transform::idct4x4(&mut current_subblock);
                add_residue(&mut y_with_border, &current_subblock, y0, x0, stride);
            }
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * stride + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * stride + x + 1];
        }

        luma_blocks
    }

    fn get_predicted_chroma_block(
        &self,
        chroma_mode: ChromaMode,
        mbx: usize,
        mby: usize,
        top_border: &[u8],
        left_border: &[u8],
    ) -> [u8; CHROMA_BLOCK_SIZE] {
        let mut chroma_with_border = create_border_chroma(mbx, mby, top_border, left_border);

        match chroma_mode {
            ChromaMode::DC => {
                predict_dcpred(
                    &mut chroma_with_border,
                    8,
                    CHROMA_STRIDE,
                    mby != 0,
                    mbx != 0,
                );
            }
            ChromaMode::V => {
                predict_vpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::H => {
                predict_hpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
            ChromaMode::TM => {
                predict_tmpred(&mut chroma_with_border, 8, 1, 1, CHROMA_STRIDE);
            }
        }

        chroma_with_border
    }

    fn get_chroma_blocks_from_predicted(
        &self,
        predicted_chroma: &[u8; CHROMA_BLOCK_SIZE],
        chroma_data: &[u8],
        mbx: usize,
        mby: usize,
    ) -> [i32; 16 * 4] {
        let mut chroma_blocks = [0i32; 16 * 4];
        let stride = CHROMA_STRIDE;

        let chroma_width = usize::from(self.macroblock_width * 8);

        for block_y in 0..2 {
            for block_x in 0..2 {
                // the index on the chroma block
                let block_index = block_y * 16 * 2 + block_x * 16;
                let border_block_index = (block_y * 4 + 1) * stride + block_x * 4 + 1;
                let chroma_data_block_index =
                    (mby * 8 + block_y * 4) * chroma_width + mbx * 8 + block_x * 4;

                let mut chroma_block = [0i32; 16];
                for y in 0..4 {
                    for x in 0..4 {
                        let predicted_index = border_block_index + y * stride + x;
                        let predicted_value = predicted_chroma[predicted_index];
                        let actual_index = chroma_data_block_index + y * chroma_width + x;
                        let actual_value = chroma_data[actual_index];
                        chroma_block[y * 4 + x] =
                            i32::from(actual_value) - i32::from(predicted_value);
                    }
                }

                transform::dct4x4(&mut chroma_block);

                chroma_blocks[block_index..][..16].copy_from_slice(&chroma_block);
            }
        }

        chroma_blocks
    }

    fn get_chroma_block_coeffs(&self, chroma_blocks: [i32; 16 * 4]) -> ChromaCoeffs {
        let mut chroma_coeffs: ChromaCoeffs = [0i32; 16 * 4];
        let segment = self.segments[0];

        for (block, coeff_block) in chroma_blocks
            .chunks_exact(16)
            .zip(chroma_coeffs.chunks_exact_mut(16))
        {
            for ((index, &value), coeff) in block.iter().enumerate().zip(coeff_block.iter_mut()) {
                let quant = if index > 0 {
                    segment.uvac
                } else {
                    segment.uvdc
                };
                *coeff = value / i32::from(quant);
            }
        }

        chroma_coeffs
    }

    fn get_dequantized_blocks_from_coeffs_chroma(
        &self,
        chroma_coeffs: &ChromaCoeffs,
    ) -> [i32; 16 * 4] {
        let mut dequantized_blocks = [0i32; 16 * 4];
        let segment = self.segments[0];

        for (coeffs_block, dequant_block) in chroma_coeffs
            .chunks_exact(16)
            .zip(dequantized_blocks.chunks_exact_mut(16))
        {
            for ((index, &coeff), dequant_value) in coeffs_block
                .iter()
                .enumerate()
                .zip(dequant_block.iter_mut())
            {
                let quant = if index > 0 {
                    segment.uvac
                } else {
                    segment.uvdc
                };
                *dequant_value = coeff * i32::from(quant);
            }

            transform::idct4x4(dequant_block);
        }

        dequantized_blocks
    }

    fn transform_chroma_blocks(
        &mut self,
        mbx: usize,
        mby: usize,
        chroma_mode: ChromaMode,
    ) -> ([i32; 16 * 4], [i32; 16 * 4]) {
        let stride = CHROMA_STRIDE;

        let mut predicted_u = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_u,
            &self.left_border_u,
        );
        let mut predicted_v = self.get_predicted_chroma_block(
            chroma_mode,
            mbx,
            mby,
            &self.top_border_v,
            &self.left_border_v,
        );

        let u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

        let u_coeffs = self.get_chroma_block_coeffs(u_blocks);
        let v_coeffs = self.get_chroma_block_coeffs(v_blocks);

        let quantized_u_residue = self.get_dequantized_blocks_from_coeffs_chroma(&u_coeffs);
        let quantized_v_residue = self.get_dequantized_blocks_from_coeffs_chroma(&v_coeffs);

        for y in 0usize..2 {
            for x in 0usize..2 {
                let i = x + y * 2;
                let urb: &[i32; 16] = quantized_u_residue[i * 16..][..16].try_into().unwrap();

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut predicted_u, urb, y0, x0, stride);

                let vrb: &[i32; 16] = quantized_v_residue[i * 16..][..16].try_into().unwrap();

                add_residue(&mut predicted_v, vrb, y0, x0, stride);
            }
        }

        // set borders
        for ((y, u_border_value), v_border_value) in self
            .left_border_u
            .iter_mut()
            .enumerate()
            .zip(self.left_border_v.iter_mut())
        {
            *u_border_value = predicted_u[y * stride + 8];
            *v_border_value = predicted_v[y * stride + 8];
        }

        for ((x, u_border_value), v_border_value) in self.top_border_u[mbx * 8..][..8]
            .iter_mut()
            .enumerate()
            .zip(self.top_border_v[mbx * 8..][..8].iter_mut())
        {
            *u_border_value = predicted_u[8 * stride + x + 1];
            *v_border_value = predicted_v[8 * stride + x + 1];
        }

        (u_blocks, v_blocks)
    }
}

fn get_coeffs0_from_block(blocks: &[i32; 16 * 16]) -> [i32; 16] {
    let mut coeffs0 = [0i32; 16];
    for (coeff, first_coeff_value) in coeffs0.iter_mut().zip(blocks.iter().step_by(16)) {
        *coeff = *first_coeff_value;
    }
    coeffs0
}

pub(crate) fn encode_frame_lossy<W: Write>(
    writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    lossy_quality: u8,
) -> Result<(), EncodingError> {
    let mut vp8_encoder = Vp8Encoder::new(writer);

    let width = width
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;
    let height = height
        .try_into()
        .map_err(|_| EncodingError::InvalidDimensions)?;

    vp8_encoder.encode_image(data, color, width, height, lossy_quality)?;

    Ok(())
}
