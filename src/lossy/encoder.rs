use std::io::Write;

use byteorder_lite::{LittleEndian, WriteBytesExt};

use super::arithmetic_encoder::{tree_encode_path, ArithmeticEncoder};
use super::common::*;
use super::loop_filter;
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
#[cfg_attr(test, derive(Debug, PartialEq))]
struct MacroblockInfo {
    luma_mode: LumaMode,
    // note ideally this would be on LumaMode::B
    // since that it's where it's valid but need to change the decoder to
    // work with that as well
    luma_bpred: Option<[IntraMode; 16]>,
    chroma_mode: ChromaMode,
    // Which of `Vp8Encoder::segments` this macroblock uses (index into
    // `segments`, 0..MAX_SEGMENTS), chosen by `classify_segments`'s activity
    // classifier and set by `choose_macroblock_info` - the only place this
    // is ever `None` is `MacroblockInfo::default()` before that runs.
    // `unwrap_or(0)` elsewhere is a defensive fallback, not a real "no
    // segment" case: segmentation is always on (see `setup_encoding`).
    segment_id: Option<usize>,

    coeffs_skipped: bool,

    /// Mirrors `Vp8Decoder`'s per-macroblock `MacroBlock::non_zero_dct`
    /// (`lossy/mod.rs`): true iff, once this macroblock is actually
    /// reconstructed, any of its luma or chroma sub-blocks has a nonzero
    /// (dequantized) coefficient. Used by `apply_loop_filter`
    /// (image-resizer#137 stage 2) exactly the way the decoder's
    /// `loop_filter` uses it - to decide `do_subblock_filtering` (15.2):
    /// `mb.luma_mode == LumaMode::B || (!mb.coeffs_skipped &&
    /// mb.non_zero_dct)`.
    ///
    /// Unlike every other field on this struct, this is *not* known at
    /// `choose_macroblock_info` time - it depends on the actual quantized
    /// residual, which only exists once `transform_luma_block` /
    /// `transform_chroma_blocks` run. `MacroblockInfo::default()` and every
    /// other place that builds one before reconstruction (`choose_
    /// macroblock_info`) leaves this `false`; `encode_image`'s real pass is
    /// the only place that fills in the real value, immediately after
    /// reconstructing each macroblock, by overwriting that macroblock's
    /// `mb_info_cache` entry.
    non_zero_dct: bool,
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
    let p = if branch_is_true {
        1.0 - p_false
    } else {
        p_false
    };
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
/// the typical energy in a block. `segment` here is always the specific
/// segment `classify_segments` assigned this macroblock to (see
/// `choose_macroblock_info`), not a frame-wide constant: adaptive
/// quantisation means different macroblocks can have different `Qstep`, so
/// `lambda` has to be recomputed per macroblock rather than once per frame.
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

// --- Loop filter level derivation --------------------------------------
//
// image-resizer#137: the encoder now applies VP8's in-loop deblocking
// filter to its own reconstruction (`Vp8Encoder::apply_loop_filter`,
// mirroring `Vp8Decoder::loop_filter`), instead of hardcoding
// `frame.filter_level` to 0 to avoid the quality loss an unconditional
// maximal filter caused (see `setup_encoding`'s doc comment on
// `Frame::filter_level` for that history). `derive_filter_level` is what
// picks the level to actually apply.

/// Empirical scale factor for `derive_filter_level`: divides the base
/// quantiser index (`quant_index`, 0..=127) to get the filter level
/// (0..=63). Picked so the coarsest quantiser (127) lands on a level well
/// short of the maximum (63) - a level of 63 unconditionally is exactly the
/// failure mode this replaces - while the finest quantisers (where blocking
/// is barely visible to begin with) stay at or near 0.
const FILTER_LEVEL_DIVISOR: u32 = 8;

/// Derives the frame-level VP8 loop filter strength (`frame.filter_level`,
/// 9.4/15) from the base luma-AC quantiser index (`quant_index`, 0..=127 -
/// the same value `setup_encoding` derives from `lossy_quality` and feeds to
/// `build_segment`).
///
/// Modelled on the general principle libvpx's own encoder uses for its
/// default (non-`--noise-sensitivity`, non-two-pass-tuned) filter level: a
/// coarser quantiser produces more visible blocking at macroblock and
/// sub-block edges, so it needs a stronger filter to smooth over it, while a
/// fine quantiser already reproduces those edges accurately and a strong
/// filter would only blur real detail. This is a plain linear mapping
/// (`quant_index / FILTER_LEVEL_DIVISOR`, clamped to the format's 0..=63
/// range) rather than libvpx's own piecewise/table-based curve - the exact
/// shape of that curve is undocumented in this codebase and not worth
/// reverse-engineering when a monotonic, order-of-magnitude-correct mapping
/// already recovers most of the benefit (see the `Frame::filter_level` doc
/// comment history this replaces: the failure mode was the level being
/// *disconnected* from the quantiser, maxed out regardless of it, not the
/// exact curve).
///
/// `FILTER_LEVEL_DIVISOR` is the one knob here, kept as a named constant so
/// it can be retuned in isolation. It has not been swept against
/// `examples/rd_eval`/`examples/ssimu2_eval` the way `LAMBDA_SCALE` and
/// `SEGMENT_QUANT_DELTAS` were (that sweep is left for the caller to run
/// against the real corpus, per this change's own instructions) - treat it
/// as a reasonable starting point, not a calibrated constant.
fn derive_filter_level(quant_index: u8) -> u8 {
    let level = u32::from(quant_index) / FILTER_LEVEL_DIVISOR;
    level.min(63) as u8
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

// --- Coefficient token probability adaptation -------------------------------
//
// A keyframe header may override any of the ~1056 coefficient token
// probabilities (`token_probs[plane][band][context][node]`,
// `TokenProbTables`) away from the `COEFF_PROBS` default, at a per-value
// cost of one flag bit plus, when set, 8 more bits (`encode_updated_token_
// probabilities`). `collect_token_counts` runs the frame once (dry, like
// `count_skipped_macroblocks`) to learn how often each branch of the
// coefficient token tree is actually taken in each bucket;
// `derive_updated_token_probs` turns those counts into the final
// `token_probs` table used both to encode the residual data and to fill in
// the header, by updating only where the measured bits saved outweigh the
// signalling cost.

/// One coefficient-token coding event within a block, as produced by
/// [`tokenize_block`]: which `(band, context)` bucket of
/// `token_probs[plane]` it reads probabilities from, the DCT token coded (a
/// `DCT_*` constant, or `DCT_EOB`), and the token tree's start index. That
/// index is `2` right after a zero coefficient (two zeros in a row can never
/// be followed by an end-of-block that the decoder couldn't have inferred by
/// just ending the block at the first one, see `write_with_tree_start_index`)
/// and `0` otherwise. `quantized_value` is the signed quantized coefficient;
/// only the real encode path needs it, to derive the sign bit and category
/// "extra" bits (both coded with fixed, non-adapted probabilities), and the
/// token-statistics pass ignores it.
struct TokenEvent {
    band: usize,
    context: usize,
    token: i8,
    start_index: usize,
    quantized_value: i32,
}

/// Quantizes, zigzags, and walks one block's coefficients, returning the
/// sequence of token-tree coding events (see `TokenEvent`) plus whether the
/// block has any non-zero coefficient (the `has_coeffs` complexity-context
/// bookkeeping `encode_residual_data` threads between blocks).
///
/// Pulled out of `encode_coefficients` so the real encode path and the
/// token-probability statistics dry run (`Vp8Encoder::collect_token_counts`)
/// tokenize every block identically - two independent copies of this logic
/// could silently drift apart, which would desync the probabilities the
/// header advertises from what the residual partitions actually use.
fn tokenize_block(
    block: &[i32; 16],
    plane: Plane,
    complexity: usize,
    dc_quant: i16,
    ac_quant: i16,
) -> (Vec<TokenEvent>, bool) {
    let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };

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

    let mut events = Vec::new();
    let mut skip_eob = false;

    for index in first_coeff..end_of_block_index {
        let coeff = zigzag_block[index];

        let band = usize::from(COEFF_BANDS[index]);
        let start_index = if skip_eob { 2 } else { 0 };

        let token = match coeff.abs() {
            0 => {
                // never going to have an end of block after a 0, so skip checking next coeff
                skip_eob = true;
                DCT_0
            }

            // just encode as literal
            literal @ 1..=4 => {
                skip_eob = false;
                literal as i8
            }

            // encode the category
            value => {
                skip_eob = false;
                match value {
                    5..=6 => DCT_CAT1,
                    7..=10 => DCT_CAT2,
                    11..=18 => DCT_CAT3,
                    19..=34 => DCT_CAT4,
                    35..=66 => DCT_CAT5,
                    67..=2048 => DCT_CAT6,
                    _ => unreachable!(),
                }
            }
        };

        events.push(TokenEvent {
            band,
            context: complexity,
            token,
            start_index,
            quantized_value: coeff,
        });

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
        events.push(TokenEvent {
            band,
            context: complexity,
            token: DCT_EOB,
            start_index: 0,
            quantized_value: 0,
        });
    }

    (events, end_of_block_index > 0)
}

/// Per-`(plane, band, context)` bucket branch counts for every internal node
/// of the coefficient token tree (`DCT_TOKEN_TREE`): `[false_count,
/// true_count]` for `token_probs[plane][band][context][node]`. Same shape as
/// `TokenProbTables`, but counting how often each branch was actually taken
/// this frame instead of holding a probability. Built by
/// `Vp8Encoder::collect_token_counts`, consumed by
/// `derive_updated_token_probs`.
type TokenCounts = [[[[[u64; 2]; NUM_DCT_TOKENS - 1]; 3]; 8]; 4];

/// Adds one block's tokenization (see `tokenize_block`) into `counts`, by
/// replaying the exact same root-to-leaf tree walk
/// `write_with_tree_start_index` performs when actually writing a token
/// (`tree_encode_path`) and counting each branch instead of encoding it.
fn accumulate_token_events(counts: &mut TokenCounts, plane: Plane, events: &[TokenEvent]) {
    for event in events {
        for (bit, prob_index) in tree_encode_path(&DCT_TOKEN_TREE, event.token, event.start_index) {
            counts[plane as usize][event.band][event.context][prob_index][usize::from(bit)] += 1;
        }
    }
}

/// Buckets with fewer than this many combined true/false observations keep
/// `COEFF_PROBS`'s default outright, without even computing a candidate
/// probability for them.
///
/// Below about 20 samples, the standard error of a binomial proportion
/// (`~1 / (2*sqrt(n))`, so +-11% at n=20) is large enough that the rounded
/// 1..=255 estimate mostly reflects which way this particular handful of
/// coefficients happened to fall, not a skew that will hold up over the rest
/// of the bucket's - mostly still-to-be-seen, since 20 is a small fraction of
/// a macroblock grid - occurrences. The bit-cost comparison in
/// `derive_updated_token_probs` is the real gatekeeper regardless (a sparse
/// bucket's total savings can't realistically outrun the flag + 8-bit
/// signalling cost), but this threshold avoids computing a confident-looking,
/// wildly extreme estimate (e.g. 255 from a 3/3 split) from a handful of
/// samples in the first place.
const MIN_TOKEN_PROB_OBSERVATIONS: u64 = 20;

/// Real entropy cost, in bits, of coding `false_count` "false" branches and
/// `true_count` "true" branches with a fixed probability `prob` (VP8
/// convention: `prob` is `P(branch is false)`, scaled to 0..=255) - i.e.
/// `false_count` uses of `branch_bit_cost(prob, false)` plus `true_count`
/// uses of `branch_bit_cost(prob, true)`, added up instead of walked one
/// branch at a time.
fn total_branch_bit_cost(false_count: u64, true_count: u64, prob: Prob) -> f64 {
    false_count as f64 * branch_bit_cost(prob, false)
        + true_count as f64 * branch_bit_cost(prob, true)
}

/// Decides, from `counts` (see `Vp8Encoder::collect_token_counts`), which of
/// the frame header's ~1056 coefficient probabilities are worth overriding
/// away from `COEFF_PROBS`, and returns the resulting table - used both to
/// fill in the header (`Vp8Encoder::encode_updated_token_probabilities`) and,
/// unchanged, as `self.token_probs` for the real encode pass, so the two
/// can never disagree.
///
/// For each bucket with enough observations (`MIN_TOKEN_PROB_OBSERVATIONS`),
/// the candidate probability is the rounded-to-nearest, clamped-to-1..=255
/// maximum-likelihood estimate from the counts (`skip_probability` uses the
/// same rounding convention, for the same reason: keep clear of the
/// unrepresentable 0/256 edges). That estimate minimises the coding cost of
/// exactly the branches observed this frame - by definition, cross-entropy
/// against an empirical distribution is minimised by coding with that same
/// distribution - so updating is never a *worse* fit for the data; the only
/// question is whether the fit is good enough to be worth paying for.
///
/// Sending the update costs the flag bit (coded `true`, at
/// `COEFF_UPDATE_PROBS`'s own probability for this node) plus a fixed 8 bits
/// for the new value; not sending it costs the flag bit alone (coded
/// `false`). Both sides are computed in real bits (`total_branch_bit_cost`,
/// `-log2(p)`), and the update is only taken when it comes out cheaper
/// overall:
///
/// ```text
/// cost(update)    = flag_cost(true)  + 8 + bits(candidate_prob)
/// cost(no update) = flag_cost(false)     + bits(default_prob)
/// update iff bits(default_prob) - bits(candidate_prob) > 8 + flag_cost(true) - flag_cost(false)
/// ```
///
/// i.e. the real bits saved on the token stream must exceed the net extra
/// cost of signalling "yes, update" instead of "no". A blanket "always
/// update" - flag cost aside - is wrong on its own merits too: for a
/// low-observation bucket the 8-bit literal alone usually costs more than
/// the tiny stream savings it buys, which is why most of the ~1056
/// probabilities are expected to stay at their default.
fn derive_updated_token_probs(counts: &TokenCounts) -> TokenProbTables {
    let mut probs = COEFF_PROBS;

    for (i, counts_i) in counts.iter().enumerate() {
        for (j, counts_j) in counts_i.iter().enumerate() {
            for (k, counts_k) in counts_j.iter().enumerate() {
                for (l, &[false_count, true_count]) in counts_k.iter().enumerate() {
                    let total = false_count + true_count;
                    if total < MIN_TOKEN_PROB_OBSERVATIONS {
                        continue;
                    }

                    let default_prob = COEFF_PROBS[i][j][k][l];
                    let scaled = (256 * false_count + total / 2) / total;
                    let candidate_prob = scaled.clamp(1, 255) as u8;

                    if candidate_prob == default_prob {
                        continue;
                    }

                    let bits_with_default =
                        total_branch_bit_cost(false_count, true_count, default_prob);
                    let bits_with_candidate =
                        total_branch_bit_cost(false_count, true_count, candidate_prob);
                    let savings = bits_with_default - bits_with_candidate;

                    let update_flag_prob = COEFF_UPDATE_PROBS[i][j][k][l];
                    let flag_cost_true = total_branch_bit_cost(0, 1, update_flag_prob);
                    let flag_cost_false = total_branch_bit_cost(1, 0, update_flag_prob);
                    let overhead = 8.0 + flag_cost_true - flag_cost_false;

                    if savings > overhead {
                        probs[i][j][k][l] = candidate_prob;
                    }
                }
            }
        }
    }

    probs
}

// --- Segmentation (adaptive quantisation) -----------------------------------
//
// VP8 keyframes can split their macroblocks across up to `MAX_SEGMENTS`
// segments (9.3), each with its own quantiser (and loop-filter level, unused
// here - see `setup_encoding`'s doc comment on `filter_level` for why the
// loop filter is off entirely). This is how an encoder spends fewer bits on
// busy, texture-masked macroblocks and more on smooth ones, where banding
// and blocking are visible: `classify_segments` decides which macroblock
// gets which segment, `SEGMENT_QUANT_DELTAS` decides what each segment's
// quantiser actually is, and `build_segment` derives the quantiser values
// from that the same way the decoder will.

fn dc_quant(index: i32) -> i16 {
    DC_QUANT[index.clamp(0, 127) as usize]
}

fn ac_quant(index: i32) -> i16 {
    AC_QUANT[index.clamp(0, 127) as usize]
}

/// Per-segment quantiser-index delta, added to the frame's base quantiser
/// index (`yac_abs`) to get that segment's actual index - see `build_segment`.
/// Ordered by `classify_segments`'s quartile rank: index 0 is the
/// lowest-activity quartile (the flattest macroblocks), `MAX_SEGMENTS - 1`
/// the busiest.
///
/// Flat regions are exactly where banding and blocking are visible, so they
/// get a negative delta (finer quantiser, more bits spent); busy/textured
/// regions mask quantisation error, so they get a positive delta (coarser
/// quantiser, fewer bits) - the same masking argument libvpx's and x264's
/// adaptive quantisation are built on. Keeping the two inner quartiles close
/// to 0 keeps the frame's *average* quantiser close to what `yac_abs` alone
/// would have given, which is what keeps `lossy_quality` monotonic with
/// segmentation on (see `encode_segment_updates`'s delta-mode doc comment).
/// Magnitudes were picked by sweeping `examples/ssimu2_eval.rs` (SSIMULACRA2
/// at fixed quality - the metric that actually decides whether a perceptual
/// reallocation like this helped, see that example's doc comment) and
/// `examples/rd_eval.rs` against the Kodak corpus; see the commit message
/// for the resulting numbers.
const SEGMENT_QUANT_DELTAS: [i8; MAX_SEGMENTS] = [-6, -2, 2, 7];

/// Builds one segment's dequantised-domain quantiser values from this
/// frame's base quantiser index and a per-segment delta, replicating
/// `Vp8Decoder::read_quantization_indices`'s formula in `lossy/mod.rs`
/// field for field - same clamps (`ac_quant`/`dc_quant`'s `0..=127` index
/// clamp, `y2ac`'s floor of 8, `uvdc`'s ceiling of 132) and all. This is
/// what makes segmentation consistent end to end: the decoder derives a
/// segment's quantiser purely from the header bytes `encode_segment_updates`
/// writes (`quantizer_level` below is exactly the `delta` that goes into
/// that header), so if this function's formula ever drifted from the
/// decoder's, the bitstream would still decode - just to the wrong pixels,
/// silently. `indices` supplies the frame-level per-plane deltas
/// (`ydc_delta` etc.), which apply identically to every segment and are
/// currently always `None`/0 - segmentation only varies `base`.
fn build_segment(yac_abs: u8, delta: i8, indices: &QuantizationIndices) -> Segment {
    let base = i32::from(yac_abs) + i32::from(delta);

    let ydc_delta = indices.ydc_delta.map_or(0, i32::from);
    let y2dc_delta = indices.y2dc_delta.map_or(0, i32::from);
    let y2ac_delta = indices.y2ac_delta.map_or(0, i32::from);
    let uvdc_delta = indices.uvdc_delta.map_or(0, i32::from);
    let uvac_delta = indices.uvac_delta.map_or(0, i32::from);

    let y2ac = ((i32::from(ac_quant(base + y2ac_delta)) * 155 / 100) as i16).max(8);
    let uvdc = dc_quant(base + uvdc_delta).min(132);

    Segment {
        ydc: dc_quant(base + ydc_delta),
        yac: ac_quant(base),
        y2dc: dc_quant(base + y2dc_delta) * 2,
        y2ac,
        uvdc,
        uvac: ac_quant(base + uvac_delta),
        delta_values: true,
        quantizer_level: delta,
        loopfilter_level: 0,
    }
}

/// Probabilities for `SEGMENT_ID_TREE` (9.3/19.2), derived from the real
/// distribution of `segment_ids` instead of the spec's 255-per-node ("almost
/// always branch false") default, which assumes segmentation is barely used
/// - the opposite of this encoder, which always segments every macroblock.
///
/// `classify_segments`'s quartile split already makes each of the tree's
/// three nodes close to a 50/50 branch by construction (each segment gets
/// about `n / 4` macroblocks), so this mostly corrects for `n` not dividing
/// evenly by 4 - but computing the exact value is one pass over `segment_ids`
/// plus three lookups, so there is no reason to settle for the approximation
/// when the real value is this cheap.
fn segment_tree_probs_for(segment_ids: &[u8]) -> [Prob; 3] {
    let mut counts = [0u32; MAX_SEGMENTS];
    for &id in segment_ids {
        counts[id as usize] += 1;
    }
    let total = segment_ids.len() as u32;

    // `SEGMENT_ID_TREE = [2, 4, -0, -1, -2, -3]`: node 0 (the root) branches
    // false to node 1 (values {0, 1}) and true to node 2 (values {2, 3});
    // node 1 branches false/true to 0/1, node 2 false/true to 2/3.
    //
    // `skip_probability(true_count, total)` computes "probability that a
    // boolean flag is false", scaled to a byte, from a count of how often
    // it's true - exactly what each node above needs, just generalised past
    // its original `coeffs_skipped` use.
    let high_half = counts[2] + counts[3];
    let prob0 = skip_probability(high_half, total);
    let prob1 = skip_probability(counts[1], counts[0] + counts[1]);
    let prob2 = skip_probability(counts[3], counts[2] + counts[3]);
    [prob0, prob1, prob2]
}

struct Vp8Encoder<W> {
    writer: W,
    frame: Frame,
    /// The encoder for the macroblock headers and the compressed frame header
    encoder: ArithmeticEncoder,
    segments: [Segment; MAX_SEGMENTS],
    segments_enabled: bool,
    /// Probabilities for `SEGMENT_ID_TREE` (9.3/19.2), derived from the
    /// actual distribution of `mb_segment_ids` by `segment_tree_probs_for`
    /// once per frame - see that function's doc comment for why the
    /// quartile classifier makes the default 128/128/128 already close to
    /// optimal, and why we compute the real value anyway.
    segment_tree_probs: [Prob; 3],
    /// Which of `segments` each macroblock uses, indexed `mby *
    /// macroblock_width + mbx`. Populated once per frame by
    /// `classify_segments` (called from `setup_encoding`, after
    /// `self.frame`/`macroblock_width`/`macroblock_height` are set) and read
    /// by `choose_macroblock_info`, which is the only place a `segment_id`
    /// is chosen for a macroblock - everywhere else threads the id that
    /// decision already made, so mode decision, quantisation and
    /// reconstruction always agree on which segment a macroblock is in.
    mb_segment_ids: Vec<u8>,

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

    /// Full-frame *unfiltered* luma reconstruction plane. Sized from the
    /// macroblock grid, `macroblock_width * 16` columns by
    /// `macroblock_height * 16` rows (row-major, stride
    /// `macroblock_width * 16`) - exactly like the decoder's own
    /// `Frame::ybuf` (`Vp8Decoder::new`, `lossy/mod.rs`), including padding
    /// past the image's real `width`/`height` for partial edge
    /// macroblocks, which is why this is sized from the macroblock grid
    /// rather than the image.
    ///
    /// Replaces the old `top_border_y`/`left_border_y` incremental caches
    /// (image-resizer#137 stage 1): `transform_luma_block` /
    /// `transform_luma_blocks_4x4` write each macroblock's reconstruction
    /// in here immediately after computing it, and
    /// `create_border_luma_from_plane` (`prediction.rs`) reads the borders
    /// for every later macroblock straight back out of it - see that
    /// function's doc comment for why raster-order encoding makes that
    /// byte-identical to the caches it replaces.
    ///
    /// The reason to keep the *plane*, not just borders, at all: the VP8
    /// loop filter (image-resizer#137 stage 2, `apply_loop_filter`) is
    /// in-loop with respect to becoming a reference/output frame, and
    /// modifies pixels up to 3 rows/columns deep on either side of a
    /// macroblock edge - far more than a 1-pixel border cache can supply.
    /// Once the plane has to exist for that, deriving borders from it is
    /// simpler than maintaining both a plane and a duplicate cache.
    recon_y: Vec<u8>,
    /// Chroma (U) counterpart of `recon_y`, sized `macroblock_width * 8` by
    /// `macroblock_height * 8`.
    recon_u: Vec<u8>,
    /// Chroma (V) counterpart of `recon_y`, sized `macroblock_width * 8` by
    /// `macroblock_height * 8`.
    recon_v: Vec<u8>,

    /// Per-macroblock RD mode decision, indexed `mby * macroblock_width +
    /// mbx`, computed once by `count_skipped_macroblocks` and reused by
    /// `collect_token_counts` *and* `encode_image`'s real pass, instead of
    /// re-running `choose_macroblock_info`'s RD search (mode search over 4
    /// luma candidates, B_PRED's 10-mode search over 16 sub-blocks, and 4
    /// chroma candidates) a second or third time for the same macroblock.
    ///
    /// Consulted by all three passes as of image-resizer#151. That relies on
    /// all three agreeing on every mode decision, macroblock for macroblock -
    /// which in turn relies on all three evolving `top_b_pred`/
    /// `left_b_pred` (the B_PRED submode entropy context
    /// `choose_macroblock_info`, via `trial_luma_bpred`'s
    /// `bpred_mode_bit_cost` rate term, reads to score candidates) exactly
    /// the same way, macroblock by macroblock. `encode_image`'s real pass
    /// advances that context inside `write_macroblock_header` (interleaved
    /// with the bits it writes, since each sub-block's write depends on the
    /// pre-update context). `count_skipped_macroblocks` and
    /// `collect_token_counts` never write bits, so they instead call
    /// `advance_bpred_context` - the same state transition, extracted so it
    /// can run without anywhere to write to - at the same point in their
    /// loop (right after the mode decision, before the transforms) that
    /// `encode_image` calls `write_macroblock_header`.
    ///
    /// Before image-resizer#151, that context update happened *only* inside
    /// `write_macroblock_header`, so it ran exclusively in the real pass:
    /// the two dry runs' `top_b_pred`/`left_b_pred` stayed frozen at
    /// `reset_frame_state`'s defaults for the whole frame. Verified
    /// empirically at the time (temporary instrumentation, since reverted)
    /// that this meant the two dry runs always agreed with each other but
    /// the real pass - whose context genuinely evolved - disagreed with them
    /// at roughly 1 in 3 macroblocks on a mixed-activity test image,
    /// including outright different winning `LumaMode`s and
    /// `coeffs_skipped` flags, not just different B_PRED submodes. That was
    /// why the cache used to stop at the two dry runs: reusing a dry-run
    /// decision in the real pass would have changed the encoded bitstream.
    /// `advance_bpred_context` closes that gap, so now the real pass's own
    /// (independently-computed, pre-#151) decisions are provably identical
    /// to what pass 1 already cached - see
    /// `all_three_passes_agree_on_mode_decisions` below - which is what
    /// makes reading the cache here safe rather than merely convenient.
    ///
    /// Reset once per frame in `setup_encoding` (not in `reset_frame_state`,
    /// which also runs *between* `count_skipped_macroblocks` and
    /// `collect_token_counts`, and again before `encode_image`'s real pass -
    /// clearing the cache there would erase pass 1's results before a later
    /// pass could read them). Sized and reallocated there too, so a second
    /// `encode_image` call on the same encoder - even at different
    /// dimensions - can never read a stale entry left over from a previous
    /// frame.
    ///
    /// `MacroblockInfo` is small (a `LumaMode`, an `Option<[IntraMode; 16]>`,
    /// a `ChromaMode`, an `Option<u8>`, a `bool`) - a few hundred KB for a
    /// whole 1080p frame's worth of macroblocks - so caching the decision
    /// itself, rather than the transform output (which stays uncached; see
    /// `transform_luma_block` / `transform_chroma_blocks`, which run fresh
    /// in every pass because they also carry border/complexity state each
    /// pass depends on), is cheap.
    mb_info_cache: Vec<Option<MacroblockInfo>>,
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
            segment_tree_probs: [128; 3],
            mb_segment_ids: Vec::new(),

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

            recon_y: Vec::new(),
            recon_u: Vec::new(),
            recon_v: Vec::new(),

            mb_info_cache: Vec::new(),
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

    /// Writes `update_segmentation()` (9.3) for this frame. Every keyframe
    /// this encoder produces stands alone (there is no previous frame's
    /// segment map to reuse, and `Vp8Decoder` only ever decodes keyframes -
    /// see its doc comment), so both `update_mb_segmentation_map` and
    /// `update_segment_feature_data` are unconditionally `true`: this
    /// frame's map and quantiser deltas are always transmitted fresh.
    fn encode_segment_updates(&mut self) {
        self.encoder.write_flag(true); // update_mb_segmentation_map
        self.encoder.write_flag(true); // update_segment_feature_data

        // segment_feature_mode: `false` selects "delta" values, added to
        // the frame's own `yac_abs` (`encode_quantization_indices`) rather
        // than replacing it outright (`true`, "absolute"). Delta mode is
        // what keeps a segment's quantiser centred on the quality the
        // caller actually asked for - `classify_segments`'s deltas widen or
        // narrow it per segment, they never override it - so `lossy_quality`
        // keeps behaving monotonically with segmentation on, matching how
        // it behaved with segmentation off.
        self.encoder.write_flag(false);

        for segment in &self.segments {
            self.encoder
                .write_optional_signed_value(7, Some(segment.quantizer_level));
        }
        for _ in 0..MAX_SEGMENTS {
            // Loop filter deltas: the frame-level filter is unconditionally
            // disabled (`frame.filter_level == 0`, see `setup_encoding`'s
            // doc comment on why), so no segment ever needs its own
            // loop-filter adjustment - `None` here costs one flag bit and
            // leaves `Segment::loopfilter_level` at its default of 0.
            self.encoder.write_optional_signed_value(6, None);
        }

        for &prob in &self.segment_tree_probs {
            // Always signal an explicit probability rather than falling
            // back to the spec's 255 default (9.3): `segment_tree_probs_for`
            // already computed the value that best matches this frame's
            // actual segment distribution, and the update itself only costs
            // 9 bits (1 flag + 8 value) per node.
            self.encoder.write_flag(true);
            self.encoder.write_literal(8, prob);
        }
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

    /// Encodes the coefficient-probability updates (9.9/13.4) for this
    /// frame's header: per `(plane, band, context, node)`, a flag for
    /// whether the probability used to encode this frame's residual data
    /// differs from the `COEFF_PROBS` default a keyframe decoder always
    /// starts from, followed by the new 8-bit value when it does.
    ///
    /// `self.token_probs` must already hold exactly the probabilities
    /// `encode_coefficients` used - set by `derive_updated_token_probs` in
    /// `encode_image`, before this is called - since whatever is written
    /// here is what the decoder will use to read every coefficient in the
    /// residual partitions. See that function's doc comment for how the
    /// per-probability update decision itself is made.
    fn encode_updated_token_probabilities(&mut self) {
        for (i, is) in COEFF_UPDATE_PROBS.iter().enumerate() {
            for (j, js) in is.iter().enumerate() {
                for (k, ks) in js.iter().enumerate() {
                    for (l, update_flag_prob) in ks.iter().enumerate() {
                        let new_prob = self.token_probs[i][j][k][l];
                        let update = new_prob != COEFF_PROBS[i][j][k][l];
                        self.encoder.write_bool(update, *update_flag_prob);
                        if update {
                            self.encoder.write_literal(8, new_prob);
                        }
                    }
                }
            }
        }
    }

    fn write_macroblock_header(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        if self.segments_enabled {
            // `encode_segment_updates` always sets `update_mb_segmentation_map`,
            // so every macroblock's id is transmitted here (11.1) - every
            // `MacroblockInfo` this encoder produces has one, since
            // `choose_macroblock_info` is the only place `segment_id` is set
            // and it always sets it from `mb_segment_ids`.
            let segment_id = macroblock_info
                .segment_id
                .expect("segments_enabled implies every macroblock has a segment_id");
            self.encoder.write_with_tree(
                &SEGMENT_ID_TREE,
                &self.segment_tree_probs,
                segment_id as i8,
            );
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

    /// The B_PRED submode entropy-context state transition that
    /// `write_macroblock_header` performs for `macroblock_info`, without any
    /// of that method's bitstream writing.
    ///
    /// `write_macroblock_header` cannot simply call this and then write the
    /// bits separately: each B_PRED sub-block's write uses
    /// `KEYFRAME_BPRED_MODE_PROBS[top][left]`, where `top`/`left` are the
    /// *pre-update* context, and the update has to happen sub-block by
    /// sub-block, interleaved with the writes, because a later sub-block in
    /// the same macroblock reads an earlier one's just-written mode as its
    /// own `top`/`left`. So this function exists purely so `count_skipped_macroblocks`
    /// / `collect_token_counts` (image-resizer#151) can advance
    /// `top_b_pred`/`left_b_pred` the same way the real pass does, without
    /// writing (or having anywhere to write) bits during a dry run - not to
    /// deduplicate `write_macroblock_header`'s own logic.
    ///
    /// This does mean the state transition is written out twice. Nothing at
    /// the type level stops the two copies from drifting apart, so
    /// `advance_bpred_context_matches_write_macroblock_header` below drives
    /// both on the same inputs and asserts the resulting
    /// `top_b_pred`/`left_b_pred` are identical - that test is what is
    /// expected to catch it if they ever do.
    fn advance_bpred_context(&mut self, macroblock_info: &MacroblockInfo, mbx: usize) {
        match macroblock_info.luma_mode.into_intra() {
            None => {
                let bpred = macroblock_info
                    .luma_bpred
                    .expect("Invalid, can't set luma mode to B without setting preds");
                for y in 0usize..4 {
                    let mut left = self.left_b_pred[y];
                    for x in 0usize..4 {
                        let intra_mode = bpred[y * 4 + x];
                        left = intra_mode;
                        self.top_b_pred[mbx * 4 + x] = intra_mode;
                    }
                    self.left_b_pred[y] = left;
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
        let (events, has_coeffs) = tokenize_block(block, plane, complexity, dc_quant, ac_quant);

        let encoder = &mut self.partitions[partition_index];
        let probs = &self.token_probs[plane as usize];

        for event in &events {
            let token_probs = &probs[event.band][event.context];
            encoder.write_with_tree_start_index(
                &DCT_TOKEN_TREE,
                token_probs,
                event.token,
                event.start_index,
            );

            if event.token == DCT_EOB || event.token == DCT_0 {
                continue;
            }

            // category tokens carry extra "which value in the category"
            // bits, coded with the fixed (never adapted) `PROB_DCT_CAT`
            // probabilities - same as `read_coefficients` in the decoder.
            if event.token >= DCT_CAT1 {
                let category = event.token;
                let category_probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];
                let value = event.quantized_value.abs();
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
            }

            // note flag means coeff is negative
            encoder.write_flag(!event.quantized_value.is_positive());
        }

        // whether the block has a non zero coefficient
        has_coeffs
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

        // Same reasoning, for the coefficient token probabilities this time:
        // `encode_compressed_frame_header` (below) has to advertise them
        // before any residual data is written, so learn the real per-bucket
        // token statistics with one more dry run and derive the frame's
        // `token_probs` from them now. `encode_coefficients` reads
        // `self.token_probs` directly, so setting it here is what makes the
        // real pass below use exactly what the header just advertised - see
        // `derive_updated_token_probs` for the update decision itself.
        let token_counts = self.collect_token_counts();
        self.reset_frame_state();
        self.token_probs = derive_updated_token_probs(&token_counts);

        self.encode_compressed_frame_header();

        // encode residual partitions first
        for mby in 0..self.macroblock_height {
            let partition_index = usize::from(mby) % self.partitions.len();
            // reset left complexity / bpreds for left of image
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];

            for mbx in 0..self.macroblock_width {
                // Reads the decision `count_skipped_macroblocks` already made
                // for this macroblock instead of calling
                // `choose_macroblock_info` a third time - safe as of
                // image-resizer#151 because `count_skipped_macroblocks` /
                // `collect_token_counts` now advance `top_b_pred`/
                // `left_b_pred` (via `advance_bpred_context`) exactly the way
                // `write_macroblock_header` below advances them, so all three
                // passes agree on every mode decision - see `mb_info_cache`'s
                // doc comment for why, and
                // `all_three_passes_agree_on_mode_decisions` for the test
                // that guards it.
                let idx = usize::from(mby) * usize::from(self.macroblock_width) + usize::from(mbx);
                let macroblock_info = self.mb_info_cache[idx].expect(
                    "count_skipped_macroblocks populates every macroblock's cached mode \
                     decision before encode_image's real pass runs",
                );

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
                let (y_block_data, luma_non_zero_dct) =
                    self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info);

                let (u_block_data, v_block_data, chroma_non_zero_dct) =
                    self.transform_chroma_blocks(mbx.into(), mby.into(), &macroblock_info);

                // Recorded for `apply_loop_filter` (image-resizer#137 stage
                // 2), which runs once over the whole frame after this loop
                // finishes and needs to know, per macroblock, whether the
                // decoder would have set `MacroBlock::non_zero_dct` - see
                // `MacroblockInfo::non_zero_dct`'s doc comment.
                self.mb_info_cache[idx] = Some(MacroblockInfo {
                    non_zero_dct: luma_non_zero_dct || chroma_non_zero_dct,
                    ..macroblock_info
                });

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

        // Every macroblock is now reconstructed (unfiltered) in `recon_y`/
        // `recon_u`/`recon_v`, and every `mb_info_cache` entry has its real
        // `non_zero_dct` filled in - exactly the precondition
        // `apply_loop_filter` documents. This mirrors `Vp8Decoder::
        // decode_frame_`'s own structure: reconstruct the whole frame
        // first, unfiltered, then filter it in one pass afterwards.
        self.apply_loop_filter();

        let compressed_header_encoder = std::mem::take(&mut self.encoder);
        let compressed_header_bytes = compressed_header_encoder.flush_and_get_buffer();

        self.write_uncompressed_frame_header(compressed_header_bytes.len() as u32)?;

        self.writer.write_all(&compressed_header_bytes)?;

        self.write_partitions()?;

        Ok(())
    }

    /// Mirrors `Vp8Decoder::calculate_filter_parameters` (`lossy/mod.rs`)
    /// exactly: given this frame's filter settings and one macroblock's
    /// segment, returns `(filter_level, interior_limit, hev_threshold)` for
    /// that macroblock - the same three values that same-named decoder
    /// method computes from the bitstream this encoder is about to emit.
    /// Any divergence here is drift between what `apply_loop_filter`
    /// applies to `recon_y`/`recon_u`/`recon_v` and what a real decoder
    /// will compute and apply from the emitted frame header.
    ///
    /// Unlike the decoder, this never reads a ref/mode delta adjustment:
    /// `self.loop_filter_adjustments` is always `false` (see its field doc
    /// comment - `encode_loop_filter_adjustments` is unimplemented and
    /// never called), matching `Vp8Decoder::loop_filter_adjustments_enabled`
    /// on every stream this encoder emits, so the decoder's corresponding
    /// branch never fires either; mirroring it here would be dead code on
    /// both sides.
    fn calculate_filter_parameters(&self, segment_id: usize) -> (u8, u8, u8) {
        let segment = self.segments[segment_id];
        let mut filter_level = i32::from(self.frame.filter_level);

        if filter_level == 0 {
            return (0, 0, 0);
        }

        if self.segments_enabled {
            if segment.delta_values {
                filter_level += i32::from(segment.loopfilter_level);
            } else {
                filter_level = i32::from(segment.loopfilter_level);
            }
        }

        let filter_level = filter_level.clamp(0, 63) as u8;

        let mut interior_limit = filter_level;
        if self.frame.sharpness_level > 0 {
            interior_limit >>= if self.frame.sharpness_level > 4 { 2 } else { 1 };
            if interior_limit > 9 - self.frame.sharpness_level {
                interior_limit = 9 - self.frame.sharpness_level;
            }
        }
        if interior_limit == 0 {
            interior_limit = 1;
        }

        let hev_threshold = if filter_level >= 40 {
            2
        } else if filter_level >= 15 {
            1
        } else {
            0
        };

        (filter_level, interior_limit, hev_threshold)
    }

    /// Applies the VP8 in-loop deblocking filter (15) to this frame's own
    /// reconstruction planes (`recon_y`/`recon_u`/`recon_v`), in place -
    /// the encoder-side mirror of `Vp8Decoder::loop_filter`, run once over
    /// the whole macroblock grid from `encode_image`, immediately after
    /// every macroblock has been reconstructed.
    ///
    /// # Why intra prediction was never at risk from this
    ///
    /// VP8's loop filter is in-loop in the sense that matters for a video
    /// codec - the filtered frame is what becomes the reference for future
    /// frames' inter prediction, and what gets displayed - but *within* one
    /// frame's own decode, intra prediction always reads unfiltered
    /// neighbours in both this encoder and the decoder: `decode_frame_`
    /// (`lossy/mod.rs`) reconstructs every macroblock first (a complete
    /// `mby`/`mbx` raster pass calling `intra_predict_luma`/
    /// `intra_predict_chroma`), and only *then* runs a second, separate
    /// raster pass calling `loop_filter` - the same two-phase structure
    /// this method and its caller now give the encoder. So the filter
    /// never had anything to do with why `filter_level` used to have to
    /// stay 0; that was purely because the encoder never *applied* the
    /// filter it was signalling (see `setup_encoding`'s doc comment on
    /// `Frame::filter_level`), not because filtering would have desynced
    /// prediction. What this method's ordering has to get right is calling
    /// `loop_filter::` in the same order, and choosing the same
    /// macroblock/subblock and simple/normal filter, `Vp8Decoder::
    /// loop_filter` does - not intra-prediction timing.
    fn apply_loop_filter(&mut self) {
        if self.frame.filter_level == 0 {
            return;
        }

        let luma_w = usize::from(self.macroblock_width) * 16;
        let chroma_w = usize::from(self.macroblock_width) * 8;

        for mby in 0..usize::from(self.macroblock_height) {
            for mbx in 0..usize::from(self.macroblock_width) {
                let idx = mby * usize::from(self.macroblock_width) + mbx;
                let info = self.mb_info_cache[idx].expect(
                    "encode_image's real pass fills in every macroblock's non_zero_dct \
                     before apply_loop_filter runs",
                );
                let segment_id = info.segment_id.unwrap_or(0);
                let (filter_level, interior_limit, hev_threshold) =
                    self.calculate_filter_parameters(segment_id);

                if filter_level == 0 {
                    continue;
                }

                let mbedge_limit = (filter_level + 2) * 2 + interior_limit;
                let sub_bedge_limit = (filter_level * 2) + interior_limit;

                // we skip subblock filtering if the coding mode isn't B_PRED and there's no DCT coefficient coded
                let do_subblock_filtering =
                    info.luma_mode == LumaMode::B || (!info.coeffs_skipped && info.non_zero_dct);

                //filter across left of macroblock
                if mbx > 0 {
                    //simple loop filtering
                    if self.frame.filter_type {
                        for y in 0usize..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16;

                            loop_filter::simple_segment_horizontal(
                                mbedge_limit,
                                &mut self.recon_y[y0 * luma_w + x0 - 4..][..8],
                            );
                        }
                    } else {
                        for y in 0usize..16 {
                            let y0 = mby * 16 + y;
                            let x0 = mbx * 16;

                            loop_filter::macroblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                mbedge_limit,
                                &mut self.recon_y[y0 * luma_w + x0 - 4..][..8],
                            );
                        }

                        for y in 0usize..8 {
                            let y0 = mby * 8 + y;
                            let x0 = mbx * 8;

                            loop_filter::macroblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                mbedge_limit,
                                &mut self.recon_u[y0 * chroma_w + x0 - 4..][..8],
                            );
                            loop_filter::macroblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                mbedge_limit,
                                &mut self.recon_v[y0 * chroma_w + x0 - 4..][..8],
                            );
                        }
                    }
                }

                //filter across vertical subblocks in macroblock
                if do_subblock_filtering {
                    if self.frame.filter_type {
                        for x in (4usize..16 - 1).step_by(4) {
                            for y in 0..16 {
                                let y0 = mby * 16 + y;
                                let x0 = mbx * 16 + x;

                                loop_filter::simple_segment_horizontal(
                                    sub_bedge_limit,
                                    &mut self.recon_y[y0 * luma_w + x0 - 4..][..8],
                                );
                            }
                        }
                    } else {
                        for x in (4usize..16 - 3).step_by(4) {
                            for y in 0..16 {
                                let y0 = mby * 16 + y;
                                let x0 = mbx * 16 + x;

                                loop_filter::subblock_filter_horizontal(
                                    hev_threshold,
                                    interior_limit,
                                    sub_bedge_limit,
                                    &mut self.recon_y[y0 * luma_w + x0 - 4..][..8],
                                );
                            }
                        }

                        for y in 0usize..8 {
                            let y0 = mby * 8 + y;
                            let x0 = mbx * 8 + 4;

                            loop_filter::subblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.recon_u[y0 * chroma_w + x0 - 4..][..8],
                            );

                            loop_filter::subblock_filter_horizontal(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.recon_v[y0 * chroma_w + x0 - 4..][..8],
                            );
                        }
                    }
                }

                //filter across top of macroblock
                if mby > 0 {
                    if self.frame.filter_type {
                        for x in 0usize..16 {
                            let y0 = mby * 16;
                            let x0 = mbx * 16 + x;

                            loop_filter::simple_segment_vertical(
                                mbedge_limit,
                                &mut self.recon_y[..],
                                y0 * luma_w + x0,
                                luma_w,
                            );
                        }
                    } else {
                        //if bottom macroblock, can only filter if there is 3 pixels below
                        for x in 0usize..16 {
                            let y0 = mby * 16;
                            let x0 = mbx * 16 + x;

                            loop_filter::macroblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                mbedge_limit,
                                &mut self.recon_y[..],
                                y0 * luma_w + x0,
                                luma_w,
                            );
                        }

                        for x in 0usize..8 {
                            let y0 = mby * 8;
                            let x0 = mbx * 8 + x;

                            loop_filter::macroblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                mbedge_limit,
                                &mut self.recon_u[..],
                                y0 * chroma_w + x0,
                                chroma_w,
                            );
                            loop_filter::macroblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                mbedge_limit,
                                &mut self.recon_v[..],
                                y0 * chroma_w + x0,
                                chroma_w,
                            );
                        }
                    }
                }

                //filter across horizontal subblock edges within the macroblock
                if do_subblock_filtering {
                    if self.frame.filter_type {
                        for y in (4usize..16 - 1).step_by(4) {
                            for x in 0..16 {
                                let y0 = mby * 16 + y;
                                let x0 = mbx * 16 + x;

                                loop_filter::simple_segment_vertical(
                                    sub_bedge_limit,
                                    &mut self.recon_y[..],
                                    y0 * luma_w + x0,
                                    luma_w,
                                );
                            }
                        }
                    } else {
                        for y in (4usize..16 - 3).step_by(4) {
                            for x in 0..16 {
                                let y0 = mby * 16 + y;
                                let x0 = mbx * 16 + x;

                                loop_filter::subblock_filter_vertical(
                                    hev_threshold,
                                    interior_limit,
                                    sub_bedge_limit,
                                    &mut self.recon_y[..],
                                    y0 * luma_w + x0,
                                    luma_w,
                                );
                            }
                        }

                        for x in 0..8 {
                            let y0 = mby * 8 + 4;
                            let x0 = mbx * 8 + x;

                            loop_filter::subblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.recon_u[..],
                                y0 * chroma_w + x0,
                                chroma_w,
                            );

                            loop_filter::subblock_filter_vertical(
                                hev_threshold,
                                interior_limit,
                                sub_bedge_limit,
                                &mut self.recon_v[..],
                                y0 * chroma_w + x0,
                                chroma_w,
                            );
                        }
                    }
                }
            }
        }
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
        // The one place a macroblock's segment id is decided - every other
        // reader of `MacroblockInfo::segment_id` (quantisation, residual
        // coding, reconstruction) just threads this same value through, so
        // they all agree on which segment - and therefore which quantiser -
        // this macroblock uses.
        let segment_id = self.mb_segment_ids[mby * usize::from(self.macroblock_width) + mbx];
        let segment = self.segments[segment_id as usize];
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
            let (distortion, rate, u_coeffs, v_coeffs) =
                self.trial_chroma(mode, mbx, mby, &segment);
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
            segment_id: Some(segment_id as usize),
            coeffs_skipped,
            // Not knowable until reconstruction - see this field's doc
            // comment. `encode_image`'s real pass fills in the true value.
            non_zero_dct: false,
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
        // This trial doesn't need the non_zero_dct flags - only the real
        // reconstruction pass (`transform_luma_block`) does, for the loop
        // filter's `do_subblock_filtering` decision.
        let (dequantized_blocks, _non_zero_dct) =
            self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs, segment);

        // Reconstruct into a copy of the predicted block so this trial never
        // touches `self.recon_y`.
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

        let mut y_with_border = create_border_luma_from_plane(mbx, mby, mbw.into(), &self.recon_y);

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
                let mut best: Option<(
                    f64,
                    IntraMode,
                    i64,
                    f64,
                    [i32; 16],
                    [u8; 16],
                    [i32; 16],
                )> = None;

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
                            let actual_value = self.frame.ybuf[y_data_block_index + y * width + x];
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
                            let actual =
                                i64::from(self.frame.ybuf[y_data_block_index + y * width + x]);
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
        segment: &Segment,
    ) -> (i64, i64, ChromaCoeffs, ChromaCoeffs) {
        let mut predicted_u = self.get_predicted_chroma_block(chroma_mode, mbx, mby, &self.recon_u);
        let mut predicted_v = self.get_predicted_chroma_block(chroma_mode, mbx, mby, &self.recon_v);

        let u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

        let u_coeffs = self.get_chroma_block_coeffs(u_blocks, segment);
        let v_coeffs = self.get_chroma_block_coeffs(v_blocks, segment);

        // This trial doesn't need the non_zero_dct flags - see the
        // equivalent note in `trial_luma_16x16`.
        let (dequantized_u, _) = self.get_dequantized_blocks_from_coeffs_chroma(&u_coeffs, segment);
        let (dequantized_v, _) = self.get_dequantized_blocks_from_coeffs_chroma(&v_coeffs, segment);

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
    fn chroma_sse(
        &self,
        recon: &[u8; CHROMA_BLOCK_SIZE],
        plane: &[u8],
        mbx: usize,
        mby: usize,
    ) -> i64 {
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
    /// than re-implementing a cheaper estimate: mode decision depends on
    /// pixel data, the pixel border state (`top_border_*`/`left_border_*`)
    /// and the B_PRED submode entropy context (`top_b_pred`/`left_b_pred`),
    /// never on coefficient entropy-coding probabilities - see
    /// `collect_token_counts`'s doc comment for why that particular
    /// distinction matters here. `advance_bpred_context` (image-resizer#151)
    /// keeps that B_PRED context evolving exactly the way
    /// `write_macroblock_header` evolves it in the real pass, at the same
    /// point in the loop (right after `choose_macroblock_info`, before the
    /// transforms) - without it, `top_b_pred`/`left_b_pred` would stay
    /// frozen at `reset_frame_state`'s defaults for the whole dry run, which
    /// made this method's mode decisions disagree with the real pass at
    /// roughly 1 in 3 macroblocks on a mixed-activity test image (measured
    /// with temporary instrumentation before the fix, since reverted).
    ///
    /// This is also where `mb_info_cache` is populated (see its doc comment
    /// on the struct): `collect_token_counts` below runs the identical dry
    /// run a second time for its own, different, statistic, and
    /// `encode_image`'s real pass after that, and both read this pass's
    /// decisions back out instead of repeating the RD search that produced
    /// them. The border/b_pred/complexity state this mutates is fully reset
    /// by `reset_frame_state` immediately afterwards.
    fn count_skipped_macroblocks(&mut self) -> (u32, u32) {
        let mut total = 0u32;
        let mut skipped = 0u32;

        for mby in 0..self.macroblock_height {
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];

            for mbx in 0..self.macroblock_width {
                let info = self.choose_macroblock_info(mbx.into(), mby.into());
                self.advance_bpred_context(&info, mbx.into());
                self.transform_luma_block(mbx.into(), mby.into(), &info);
                self.transform_chroma_blocks(mbx.into(), mby.into(), &info);

                let idx = usize::from(mby) * usize::from(self.macroblock_width) + usize::from(mbx);
                self.mb_info_cache[idx] = Some(info);

                total += 1;
                if info.coeffs_skipped {
                    skipped += 1;
                }
            }
        }

        (skipped, total)
    }

    /// Dry run of the whole frame's macroblock loop, counting how often each
    /// branch of the coefficient token tree is taken per `(plane, band,
    /// context)` bucket (`TokenCounts`) - the statistics
    /// `derive_updated_token_probs` turns into the frame header's
    /// coefficient probability updates.
    ///
    /// Same two-pass shape as `count_skipped_macroblocks` just above, over
    /// the same macroblock grid, starting from the same `reset_frame_state`
    /// defaults, mutating `top_border_*`/`left_border_*`/complexity/
    /// `top_b_pred`/`left_b_pred` exactly the same way given the same
    /// sequence of `MacroblockInfo` (both call `advance_bpred_context` right
    /// after the mode decision, same as `count_skipped_macroblocks` does -
    /// see that method's doc comment). That means this method's mode
    /// decisions are byte-identical to that method's, macroblock for
    /// macroblock, by induction on the (matching) border/context state -
    /// which is exactly why this reads `mb_info_cache` (filled by
    /// `count_skipped_macroblocks`, immediately before this runs) instead of
    /// calling `choose_macroblock_info` again: it would just recompute the
    /// same answer.
    ///
    /// As of image-resizer#151 this equivalence extends to `encode_image`'s
    /// real pass too (see `mb_info_cache`'s doc comment on the struct for
    /// why), which is why that pass also reads from the cache instead of
    /// calling `choose_macroblock_info` a third time. This method still does
    /// not itself write into the cache; it only ever consumes what
    /// `count_skipped_macroblocks` already produced for this frame. The
    /// border/b_pred/complexity state this mutates is fully reset by
    /// `reset_frame_state` immediately afterwards, same as after
    /// `count_skipped_macroblocks`.
    fn collect_token_counts(&mut self) -> TokenCounts {
        let mut counts: TokenCounts = [[[[[0u64; 2]; NUM_DCT_TOKENS - 1]; 3]; 8]; 4];

        for mby in 0..self.macroblock_height {
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];

            for mbx in 0..self.macroblock_width {
                let mbx = usize::from(mbx);
                let mby = usize::from(mby);
                let idx = mby * usize::from(self.macroblock_width) + mbx;
                let info = self.mb_info_cache[idx].expect(
                    "count_skipped_macroblocks populates every macroblock's cached mode \
                     decision before collect_token_counts runs",
                );
                self.advance_bpred_context(&info, mbx);

                let (y_block_data, _) = self.transform_luma_block(mbx, mby, &info);
                let (u_block_data, v_block_data, _) = self.transform_chroma_blocks(mbx, mby, &info);

                if info.coeffs_skipped {
                    // matches `encode_image`'s handling of a skipped
                    // macroblock: no residual data (and so no tokens) is
                    // ever coded for it, but the complexity context it
                    // leaves for its neighbours is still all-zero.
                    self.left_complexity.clear(info.luma_mode != LumaMode::B);
                    self.top_complexity[mbx].clear(info.luma_mode != LumaMode::B);
                    continue;
                }

                self.accumulate_residual_token_counts(
                    &info,
                    mbx,
                    &y_block_data,
                    &u_block_data,
                    &v_block_data,
                    &mut counts,
                );
            }
        }

        counts
    }

    /// Tokenizes and counts one non-skipped macroblock's residual data into
    /// `counts` - the statistics-only counterpart of `encode_residual_data`,
    /// which this mirrors block for block (Y2, then the 16 luma sub-blocks,
    /// then chroma), including its left/top complexity-context threading,
    /// so the `(band, context)` bucket each token is counted into is exactly
    /// the one `encode_coefficients` will use for that same coefficient in
    /// the real pass.
    fn accumulate_residual_token_counts(
        &mut self,
        macroblock_info: &MacroblockInfo,
        mbx: usize,
        y_block_data: &[i32; 16 * 16],
        u_block_data: &[i32; 16 * 4],
        v_block_data: &[i32; 16 * 4],
        counts: &mut TokenCounts,
    ) {
        let mut plane = if macroblock_info.luma_mode == LumaMode::B {
            Plane::YCoeff0
        } else {
            Plane::Y2
        };

        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        if plane == Plane::Y2 {
            let mut coeffs0 = get_coeffs0_from_block(y_block_data);
            transform::wht4x4(&mut coeffs0);

            let complexity = self.left_complexity.y2 + self.top_complexity[mbx].y2;
            let (events, has_coeffs) = tokenize_block(
                &coeffs0,
                Plane::Y2,
                complexity.into(),
                segment.y2dc,
                segment.y2ac,
            );
            accumulate_token_events(counts, Plane::Y2, &events);

            self.left_complexity.y2 = if has_coeffs { 1 } else { 0 };
            self.top_complexity[mbx].y2 = if has_coeffs { 1 } else { 0 };

            plane = Plane::YCoeff1;
        }

        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let block = y_block_data[y * 4 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].y[x];
                let complexity = left + top;

                let (events, has_coeffs) =
                    tokenize_block(block, plane, complexity.into(), segment.ydc, segment.yac);
                accumulate_token_events(counts, plane, &events);

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.y[y] = left;
        }

        plane = Plane::Chroma;

        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let block = u_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].u[x];
                let complexity = left + top;

                let (events, has_coeffs) =
                    tokenize_block(block, plane, complexity.into(), segment.uvdc, segment.uvac);
                accumulate_token_events(counts, plane, &events);

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.u[y] = left;
        }

        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let block = v_block_data[y * 2 * 16 + x * 16..][..16]
                    .try_into()
                    .unwrap();

                let top = self.top_complexity[mbx].v[x];
                let complexity = left + top;

                let (events, has_coeffs) =
                    tokenize_block(block, plane, complexity.into(), segment.uvdc, segment.uvac);
                accumulate_token_events(counts, plane, &events);

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };
            }
            self.left_complexity.v[y] = left;
        }
    }

    /// Population variance of one macroblock's source luma (16x16 = 256
    /// samples) - the standard, cheap proxy for local activity used by
    /// `classify_segments` to decide how much quantisation error a
    /// macroblock can absorb before it becomes visible. Reads
    /// `self.frame.ybuf` directly (source pixels, not any prediction), so it
    /// is available as soon as `self.frame` is set, independent of mode
    /// decision order.
    fn macroblock_luma_variance(&self, mbx: usize, mby: usize) -> f64 {
        let width = usize::from(self.macroblock_width) * 16;
        let mut sum: i64 = 0;
        let mut sum_sq: i64 = 0;
        for y in 0..16 {
            let row_start = (mby * 16 + y) * width + mbx * 16;
            for x in 0..16 {
                let v = i64::from(self.frame.ybuf[row_start + x]);
                sum += v;
                sum_sq += v * v;
            }
        }
        const N: f64 = 256.0;
        let mean = sum as f64 / N;
        let mean_sq = sum_sq as f64 / N;
        // Rounding can push this fractionally below 0 for a perfectly flat
        // block; clamp rather than let a negative "variance" through.
        (mean_sq - mean * mean).max(0.0)
    }

    /// Assigns every macroblock to one of `MAX_SEGMENTS` segments by the
    /// quartile of its source-luma variance (`macroblock_luma_variance`)
    /// among this frame's macroblocks: segment 0 is the flattest quarter,
    /// `MAX_SEGMENTS - 1` the busiest. Returns the per-macroblock segment
    /// id, indexed `mby * macroblock_width + mbx` (the same indexing
    /// `mb_segment_ids` is stored and read with).
    ///
    /// Quartiles over k-means: k-means needs an iterative fit to converge on
    /// centroids, and even then can land an uneven number of macroblocks per
    /// cluster depending on the frame's variance distribution. A single sort
    /// gives an exactly-even split by construction, in one pass,
    /// deterministically - and since `MAX_SEGMENTS` is fixed at 4 by the
    /// bitstream format (9.3) rather than chosen per frame, there is no
    /// "natural cluster count" here for k-means to discover that a fixed
    /// quartile split doesn't already give just as well. An even split also
    /// has a convenient side effect `segment_tree_probs_for` relies on: it
    /// keeps each segment-id tree node close to a 50/50 branch regardless of
    /// the image's actual variance distribution.
    fn classify_segments(&self) -> Vec<u8> {
        let mb_w = usize::from(self.macroblock_width);
        let mb_h = usize::from(self.macroblock_height);
        let n = mb_w * mb_h;

        let variances: Vec<f64> = (0..n)
            .map(|i| self.macroblock_luma_variance(i % mb_w, i / mb_w))
            .collect();

        let mut by_variance: Vec<usize> = (0..n).collect();
        by_variance.sort_by(|&a, &b| variances[a].total_cmp(&variances[b]));

        let mut segment_ids = vec![0u8; n];
        for (rank, &mb_index) in by_variance.iter().enumerate() {
            let quartile = rank * MAX_SEGMENTS / n.max(1);
            segment_ids[mb_index] = quartile.min(MAX_SEGMENTS - 1) as u8;
        }
        segment_ids
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

        // choosing the quantization quality based on the quality passed in
        if lossy_quality > 100 {
            panic!("lossy quality must be between 0 and 100");
        }

        let quant_index: u8 = (127 - u16::from(lossy_quality) * 127 / 100) as u8;

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
            // image-resizer#137: derived from the same quantiser index
            // every segment's own quantiser is built from
            // (`derive_filter_level`), then actually applied to the
            // encoder's own reconstruction by `apply_loop_filter` (called
            // from `encode_image`, after every macroblock has been
            // reconstructed - mirroring `Vp8Decoder::loop_filter`'s own
            // separate, whole-frame pass) before this frame's bitstream is
            // finalised. Previously hardcoded to 0: the loop filter is
            // in-loop with respect to becoming a reference/display frame
            // (see `apply_loop_filter`'s doc comment for why that does
            // *not* mean intra prediction itself was ever at risk here),
            // and this encoder used to never apply it to its own
            // reconstruction at all, so any nonzero level it signalled
            // was purely a post-processing blur the RD search never
            // accounted for - measured to *cost* size at matched quality
            // (examples/rd_eval.rs, six Kodak images, three DSSIM targets:
            // disabling it entirely bought -4.0% mean / -3.8% median size,
            // worse on 0 of 18 points). Actually applying the filter here
            // is what makes signalling a nonzero level correct rather than
            // just less-wrong.
            filter_level: derive_filter_level(quant_index),
            sharpness_level: 7,
        };

        self.token_probs = COEFF_PROBS;

        self.quantization_indices = QuantizationIndices {
            yac_abs: quant_index,
            ..Default::default()
        };

        // Adaptive quantisation (VP8 segmentation, 9.3): classify every
        // macroblock by source-luma activity, then give each of the 4
        // resulting segments its own quantiser via `SEGMENT_QUANT_DELTAS`.
        // `self.frame` (source pixels) and `macroblock_width`/`_height` are
        // already set above, which is everything `classify_segments` needs.
        //
        // This is unconditional - every frame this encoder produces is
        // segmented - rather than gated behind a flag: it is strictly more
        // expressive than the single-quantiser path it replaces (all 4
        // segments collapse to the same quantiser if `SEGMENT_QUANT_DELTAS`
        // were all 0), and there is no reason a caller would want the coarser
        // behaviour on purpose.
        self.segments_enabled = true;
        self.mb_segment_ids = self.classify_segments();
        for (segment, &delta) in self.segments.iter_mut().zip(SEGMENT_QUANT_DELTAS.iter()) {
            *segment = build_segment(quant_index, delta, &self.quantization_indices);
        }
        self.segment_tree_probs = segment_tree_probs_for(&self.mb_segment_ids);

        // Fresh, all-`None` per frame - see `mb_info_cache`'s doc comment for
        // why this lives here rather than in `reset_frame_state` (which also
        // runs *between* the three passes that share this cache, and would
        // erase it mid-frame if it reset it too). Reallocating by the
        // current `mb_width`/`mb_height` here, on every `encode_image` call,
        // is also what guarantees a second call on the same encoder - even
        // at different image dimensions - can never read a stale entry left
        // over from a previous frame.
        self.mb_info_cache = vec![None; usize::from(mb_width) * usize::from(mb_height)];

        self.reset_frame_state();
    }

    /// Resets every piece of per-frame encoding state that prediction reads
    /// (borders, B_PRED context, coefficient complexity) back to its
    /// initial, "no macroblocks encoded yet" values. Called from
    /// `setup_encoding` (first use) and from `encode_image` after each of
    /// its two dry runs - `count_skipped_macroblocks` and
    /// `collect_token_counts` - both of which mutate all of this exactly
    /// like the real pass would, and must not leak into it, into each
    /// other, or into the real pass that follows.
    fn reset_frame_state(&mut self) {
        let mb_width = self.macroblock_width;
        let mb_height = self.macroblock_height;

        self.top_complexity = vec![Complexity::default(); usize::from(mb_width)];
        self.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        self.left_b_pred = [IntraMode::default(); 4];

        // Every read of these planes (via `create_border_luma_from_plane` /
        // `create_border_chroma_from_plane`) is gated by an `mbx == 0` /
        // `mby == 0` check that never touches the plane at all, so the
        // fill value here is never actually observed - see those
        // functions' doc comments. Zero-filling (rather than 127/129, the
        // old caches' reset values) is simplest and makes an accidental
        // unguarded read obviously wrong instead of silently plausible.
        self.recon_y = vec![0u8; usize::from(mb_width) * 16 * usize::from(mb_height) * 16];
        self.recon_u = vec![0u8; usize::from(mb_width) * 8 * usize::from(mb_height) * 8];
        self.recon_v = vec![0u8; usize::from(mb_width) * 8 * usize::from(mb_height) * 8];
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

        let mut y_with_border = create_border_luma_from_plane(mbx, mby, mbw.into(), &self.recon_y);

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

    /// Also returns, per luma sub-block, whether it contributes a nonzero
    /// coefficient to `MacroblockInfo::non_zero_dct` - mirroring
    /// `Vp8Decoder::read_residual_data`'s `if block[0] != 0 || n` check
    /// (`lossy/mod.rs`) for the Y2 case: `block[0]` there is exactly
    /// `coeffs.y2_coeffs[k]` after the IWHT below (the decoder dequantizes
    /// then IWHTs Y2 the same way), and `n` is "this sub-block has a
    /// nonzero AC coefficient", checked here on `luma_block[1..]` right
    /// after dequantizing it (and before the IDCT below turns it from a
    /// frequency- into a spatial-domain block) - dequantizing never changes
    /// zero-ness, since `segment.yac`/`y2ac`/`y2dc` are always nonzero.
    fn get_dequantized_blocks_from_coeffs_luma_16x16(
        &self,
        coeffs: &mut Luma16x16Coeffs,
        segment: &Segment,
    ) -> ([i32; 16 * 16], [bool; 16]) {
        let mut dequantized_luma_residue = [0i32; 16 * 16];
        let mut non_zero_dct = [false; 16];

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

            non_zero_dct[k] = coeffs.y2_coeffs[k] != 0 || luma_block[1..].iter().any(|&v| v != 0);

            luma_block[0] = coeffs.y2_coeffs[k];

            transform::idct4x4(luma_block);

            dequantized_luma_residue[k * 16..][..16].copy_from_slice(luma_block);
        }

        (dequantized_luma_residue, non_zero_dct)
    }

    // Transforms the luma macroblock in the following ways
    // 1. Does the luma prediction and subtracts from the block
    // 2. Converts the block so each 4x4 subblock is contiguous within the block
    // 3. Does the DCT on each subblock
    // 4. Quantizes the block and dequantizes each subblock
    // 5. Calculates the quantized block - this can be used to calculate how accurate the
    // result is and is used to populate the borders for the next macroblock
    //
    // Segment consistency: this always looks up `self.segments[macroblock_info
    // .segment_id]` - the same id `choose_macroblock_info` picked for this
    // macroblock via `classify_segments` - rather than any fixed segment.
    // `choose_macroblock_info`'s RD search already priced every mode against
    // that same segment's quantiser (`mode_decision_lambda`,
    // `trial_luma_16x16`/`trial_luma_bpred`), so reconstruction has to use
    // the identical quantiser too: any mismatch here would still decode
    // (the bitstream itself is self-consistent), but the border pixels this
    // writes for later macroblocks to predict from would silently diverge
    // from what a real decoder reconstructs, since the decoder always
    // dequantizes with the segment id actually written in the header.
    ///
    /// Also returns `non_zero_dct` (see `MacroblockInfo::non_zero_dct`'s
    /// doc comment): whether any luma sub-block of this macroblock has a
    /// nonzero coefficient, mirroring `Vp8Decoder::read_residual_data`'s
    /// `mb.non_zero_dct`.
    fn transform_luma_block(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> ([i32; 16 * 16], bool) {
        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        if macroblock_info.luma_mode == LumaMode::B {
            if let Some(bpred_modes) = macroblock_info.luma_bpred {
                return self.transform_luma_blocks_4x4(bpred_modes, mbx, mby, &segment);
            } else {
                panic!("Invalid, need bpred modes for luma mode B");
            }
        }

        let mut y_with_border =
            self.get_predicted_luma_block_16x16(macroblock_info.luma_mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&y_with_border, mbx, mby);

        // get coeffs
        let mut coeffs = self.get_luma_block_coeffs_16x16(luma_blocks, &segment);

        // now we're essentially applying the same functions as the decoder in order to ensure
        // that the border is the same as the one used for the decoder in the same macroblock
        let (dequantized_blocks, non_zero_dct_per_block) =
            self.get_dequantized_blocks_from_coeffs_luma_16x16(&mut coeffs, &segment);
        let non_zero_dct = non_zero_dct_per_block.iter().any(|&v| v);

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

        self.write_luma_recon(mbx, mby, &y_with_border);

        (luma_blocks, non_zero_dct)
    }

    /// Writes a just-reconstructed macroblock's 16x16 interior (i.e.
    /// excluding the 1-pixel-plus border it was predicted from) from
    /// `y_with_border` into `recon_y`, at that macroblock's position in the
    /// full-frame plane - see `recon_y`'s doc comment. Replaces the old
    /// per-macroblock `left_border_y`/`top_border_y` cache updates; shared
    /// by `transform_luma_block` (16x16 modes) and
    /// `transform_luma_blocks_4x4` (B_PRED), which both reconstruct into
    /// the same `[u8; LUMA_BLOCK_SIZE]` border-buffer shape (stride
    /// `LUMA_STRIDE`) before calling this.
    fn write_luma_recon(&mut self, mbx: usize, mby: usize, y_with_border: &[u8; LUMA_BLOCK_SIZE]) {
        let luma_width = usize::from(self.macroblock_width) * 16;
        for y in 0..16 {
            let src = (1 + y) * LUMA_STRIDE + 1;
            let dst = (mby * 16 + y) * luma_width + mbx * 16;
            self.recon_y[dst..dst + 16].copy_from_slice(&y_with_border[src..src + 16]);
        }
    }

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
        segment: &Segment,
    ) -> ([i32; 16 * 16], bool) {
        let mut luma_blocks = [0i32; 16 * 16];
        let stride = 1usize + 16 + 4;
        let mbw = self.macroblock_width;
        let width = usize::from(mbw * 16);

        let mut y_with_border = create_border_luma_from_plane(mbx, mby, mbw.into(), &self.recon_y);

        // Mirrors `Vp8Decoder::read_residual_data`'s `mb.non_zero_dct`
        // (`lossy/mod.rs`): true iff any of this macroblock's 16 luma
        // sub-blocks has a nonzero (dequantized, pre-IDCT) coefficient -
        // B_PRED has no separate Y2 plane, so unlike the 16x16 path below
        // every sub-block's own index 0 is a real coded DC term, not a
        // placeholder. Dequantizing never changes zero-ness (multiplying by
        // the always-nonzero `quant` factor), so checking the
        // quantized-then-dequantized value is equivalent to checking the
        // quantized value the decoder actually reads.
        let mut non_zero_dct = false;

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

                if current_subblock.iter().any(|&v| v != 0) {
                    non_zero_dct = true;
                }

                transform::idct4x4(&mut current_subblock);
                add_residue(&mut y_with_border, &current_subblock, y0, x0, stride);
            }
        }

        self.write_luma_recon(mbx, mby, &y_with_border);

        (luma_blocks, non_zero_dct)
    }

    fn get_predicted_chroma_block(
        &self,
        chroma_mode: ChromaMode,
        mbx: usize,
        mby: usize,
        recon_plane: &[u8],
    ) -> [u8; CHROMA_BLOCK_SIZE] {
        let mbw = self.macroblock_width;
        let mut chroma_with_border =
            create_border_chroma_from_plane(mbx, mby, mbw.into(), recon_plane);

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

    fn get_chroma_block_coeffs(
        &self,
        chroma_blocks: [i32; 16 * 4],
        segment: &Segment,
    ) -> ChromaCoeffs {
        let mut chroma_coeffs: ChromaCoeffs = [0i32; 16 * 4];

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

    /// Also returns, per 4x4 chroma block, whether it contributes a
    /// nonzero coefficient to `MacroblockInfo::non_zero_dct` - mirroring
    /// `Vp8Decoder::read_residual_data`'s `if block[0] != 0 || n` check for
    /// the `Plane::Chroma` case (`lossy/mod.rs`): unlike luma's Y2 split,
    /// chroma has no separate DC plane, so index 0 here is a real coded DC
    /// term like every other index - checked (dequantized, pre-IDCT)
    /// exactly like `get_dequantized_blocks_from_coeffs_luma_16x16`'s AC
    /// check, for the same reason (dequantizing never changes zero-ness).
    fn get_dequantized_blocks_from_coeffs_chroma(
        &self,
        chroma_coeffs: &ChromaCoeffs,
        segment: &Segment,
    ) -> ([i32; 16 * 4], [bool; 4]) {
        let mut dequantized_blocks = [0i32; 16 * 4];
        let mut non_zero_dct = [false; 4];

        for ((coeffs_block, dequant_block), non_zero) in chroma_coeffs
            .chunks_exact(16)
            .zip(dequantized_blocks.chunks_exact_mut(16))
            .zip(non_zero_dct.iter_mut())
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

            *non_zero = dequant_block.iter().any(|&v| v != 0);

            transform::idct4x4(dequant_block);
        }

        (dequantized_blocks, non_zero_dct)
    }

    /// Also returns `non_zero_dct` (see `MacroblockInfo::non_zero_dct`'s
    /// doc comment): whether any of the 4 U or 4 V chroma sub-blocks of
    /// this macroblock has a nonzero coefficient.
    fn transform_chroma_blocks(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
    ) -> ([i32; 16 * 4], [i32; 16 * 4], bool) {
        let stride = CHROMA_STRIDE;
        let chroma_mode = macroblock_info.chroma_mode;
        // Same segment `choose_macroblock_info` picked for this macroblock
        // (via `macroblock_info.segment_id`) - see `transform_luma_block`'s
        // doc comment for why reconstruction has to agree with mode
        // decision on this.
        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];

        let mut predicted_u = self.get_predicted_chroma_block(chroma_mode, mbx, mby, &self.recon_u);
        let mut predicted_v = self.get_predicted_chroma_block(chroma_mode, mbx, mby, &self.recon_v);

        let u_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_u, &self.frame.ubuf, mbx, mby);
        let v_blocks =
            self.get_chroma_blocks_from_predicted(&predicted_v, &self.frame.vbuf, mbx, mby);

        let u_coeffs = self.get_chroma_block_coeffs(u_blocks, &segment);
        let v_coeffs = self.get_chroma_block_coeffs(v_blocks, &segment);

        let (quantized_u_residue, u_non_zero) =
            self.get_dequantized_blocks_from_coeffs_chroma(&u_coeffs, &segment);
        let (quantized_v_residue, v_non_zero) =
            self.get_dequantized_blocks_from_coeffs_chroma(&v_coeffs, &segment);
        let non_zero_dct = u_non_zero.iter().any(|&v| v) || v_non_zero.iter().any(|&v| v);

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

        self.write_chroma_recon(mbx, mby, &predicted_u, &predicted_v);

        (u_blocks, v_blocks, non_zero_dct)
    }

    /// Writes a just-reconstructed macroblock's 8x8 chroma interior from
    /// `predicted_u`/`predicted_v` (each still carrying the 1-pixel border
    /// they were predicted from) into `recon_u`/`recon_v`, at that
    /// macroblock's position in the full-frame planes - see `recon_y`'s
    /// doc comment (the chroma planes are sized and indexed the same way,
    /// just at 8x8 instead of 16x16 per macroblock). Replaces the old
    /// per-macroblock `left_border_u`/`left_border_v`/`top_border_u`/
    /// `top_border_v` cache updates.
    fn write_chroma_recon(
        &mut self,
        mbx: usize,
        mby: usize,
        predicted_u: &[u8; CHROMA_BLOCK_SIZE],
        predicted_v: &[u8; CHROMA_BLOCK_SIZE],
    ) {
        let chroma_width = usize::from(self.macroblock_width) * 8;
        let stride = CHROMA_STRIDE;
        for y in 0..8 {
            let src = (1 + y) * stride + 1;
            let dst = (mby * 8 + y) * chroma_width + mbx * 8;
            self.recon_u[dst..dst + 8].copy_from_slice(&predicted_u[src..src + 8]);
            self.recon_v[dst..dst + 8].copy_from_slice(&predicted_v[src..src + 8]);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder_with_top_context(mb_width: u16) -> Vp8Encoder<Vec<u8>> {
        let mut encoder = Vp8Encoder::new(Vec::new());
        encoder.macroblock_width = mb_width;
        encoder.top_b_pred = vec![IntraMode::default(); 4 * usize::from(mb_width)];
        encoder.left_b_pred = [IntraMode::default(); 4];
        encoder
    }

    /// A handful of `MacroblockInfo` shapes covering both arms of the
    /// `luma_mode.into_intra()` match in `write_macroblock_header` /
    /// `advance_bpred_context`: whole-macroblock intra modes (`Some`, via
    /// `LumaMode::DC`/`TM`) and B_PRED (`None`, via `LumaMode::B`) with a mix
    /// of uniform and varied submodes.
    fn sample_macroblock_infos() -> Vec<MacroblockInfo> {
        vec![
            MacroblockInfo {
                luma_mode: LumaMode::DC,
                luma_bpred: None,
                chroma_mode: ChromaMode::DC,
                segment_id: Some(0),
                coeffs_skipped: false,
                non_zero_dct: false,
            },
            MacroblockInfo {
                luma_mode: LumaMode::TM,
                luma_bpred: None,
                chroma_mode: ChromaMode::V,
                segment_id: Some(1),
                coeffs_skipped: true,
                non_zero_dct: false,
            },
            MacroblockInfo {
                luma_mode: LumaMode::B,
                luma_bpred: Some([
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
                    IntraMode::DC,
                    IntraMode::TM,
                    IntraMode::VE,
                    IntraMode::HE,
                    IntraMode::LD,
                    IntraMode::RD,
                ]),
                chroma_mode: ChromaMode::H,
                segment_id: Some(2),
                coeffs_skipped: false,
                non_zero_dct: false,
            },
            MacroblockInfo {
                luma_mode: LumaMode::B,
                luma_bpred: Some([IntraMode::HU; 16]),
                chroma_mode: ChromaMode::TM,
                segment_id: Some(3),
                coeffs_skipped: false,
                non_zero_dct: false,
            },
            MacroblockInfo {
                luma_mode: LumaMode::H,
                luma_bpred: None,
                chroma_mode: ChromaMode::DC,
                segment_id: Some(0),
                coeffs_skipped: false,
                non_zero_dct: false,
            },
        ]
    }

    /// Regression guard for the intentional duplication `advance_bpred_context`'s
    /// doc comment calls out: `write_macroblock_header` updates
    /// `top_b_pred`/`left_b_pred` interleaved with its bitstream writes,
    /// while `advance_bpred_context` performs the identical state transition
    /// with no writes, for the dry runs. Nothing at the type level keeps
    /// these in sync, so this drives both, on the same sequence of
    /// `MacroblockInfo` and across several `mbx` positions, and asserts the
    /// resulting `top_b_pred`/`left_b_pred` never diverge.
    #[test]
    fn advance_bpred_context_matches_write_macroblock_header() {
        let mb_width = 6u16;

        for mbx in 0..usize::from(mb_width) {
            let mut via_write = encoder_with_top_context(mb_width);
            let mut via_advance = encoder_with_top_context(mb_width);

            for info in sample_macroblock_infos() {
                via_write.write_macroblock_header(&info, mbx);
                via_advance.advance_bpred_context(&info, mbx);

                assert_eq!(
                    via_write.top_b_pred, via_advance.top_b_pred,
                    "top_b_pred diverged for mbx={mbx}"
                );
                assert_eq!(
                    via_write.left_b_pred, via_advance.left_b_pred,
                    "left_b_pred diverged for mbx={mbx}"
                );
            }
        }
    }

    /// Deterministic pseudo-random byte source (xorshift32) for building
    /// small, reproducible pixel fixtures without pulling in `rand` - all
    /// this needs is "not flat", not real randomness quality.
    struct Xorshift32(u32);

    impl Xorshift32 {
        fn next_u8(&mut self) -> u8 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            (x & 0xff) as u8
        }
    }

    fn flat_rgb(width: u16, height: u16) -> Vec<u8> {
        let mut pixels = vec![0u8; usize::from(width) * usize::from(height) * 3];
        for chunk in pixels.chunks_exact_mut(3) {
            chunk.copy_from_slice(&[60, 120, 180]);
        }
        pixels
    }

    fn noisy_rgb(width: u16, height: u16) -> Vec<u8> {
        let mut rng = Xorshift32(0x2222_1111 ^ (u32::from(width) << 16) ^ u32::from(height));
        let mut pixels = vec![0u8; usize::from(width) * usize::from(height) * 3];
        for b in pixels.iter_mut() {
            *b = rng.next_u8();
        }
        pixels
    }

    /// Mixed low/high activity, half flat gradient and half checkerboard, so
    /// `classify_segments` produces more than one segment and B_PRED has a
    /// real shot at winning mode decision in at least part of the frame.
    fn mixed_rgb(width: u16, height: u16) -> Vec<u8> {
        let mut pixels = vec![0u8; usize::from(width) * usize::from(height) * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (usize::from(y) * usize::from(width) + usize::from(x)) * 3;
                let (r, g, b) = if x < width / 2 {
                    let v = (u32::from(x) * 255 / u32::from(width.max(1))) as u8;
                    (v, v, v)
                } else if (x / 4 + y / 4) % 2 == 0 {
                    (20, 20, 20)
                } else {
                    (235, 235, 235)
                };
                pixels[i] = r;
                pixels[i + 1] = g;
                pixels[i + 2] = b;
            }
        }
        pixels
    }

    /// Runs an independent re-implementation of `encode_image`'s real,
    /// bit-writing macroblock loop - calling `choose_macroblock_info` (and
    /// `write_macroblock_header`, `transform_luma_block`,
    /// `transform_chroma_blocks`, `encode_residual_data`) directly, the way
    /// that loop did before image-resizer#151 started reading
    /// `mb_info_cache` - and returns the `MacroblockInfo` it independently
    /// decided on for every macroblock, in `mby`-major, `mbx`-minor order.
    ///
    /// This deliberately does not call `encode_image` itself: after #151,
    /// `encode_image`'s loop just reads `mb_info_cache`, so calling it would
    /// only compare the cache against itself. This function exists so the
    /// test below has a decision that was, genuinely, computed independently
    /// of the cache to compare the cache against.
    fn independent_real_pass_infos(encoder: &mut Vp8Encoder<Vec<u8>>) -> Vec<MacroblockInfo> {
        let mut infos = Vec::new();

        for mby in 0..encoder.macroblock_height {
            encoder.left_complexity = Complexity::default();
            encoder.left_b_pred = [IntraMode::default(); 4];

            for mbx in 0..encoder.macroblock_width {
                let info = encoder.choose_macroblock_info(mbx.into(), mby.into());
                encoder.write_macroblock_header(&info, mbx.into());

                // `non_zero_dct` deliberately left at its default `false`
                // here (discarding both transform calls' second/third
                // return values): it is not a mode-decision field (it can
                // only be known after reconstruction, see its doc comment),
                // so - like pass 1 (`count_skipped_macroblocks`) and pass 2
                // (`collect_token_counts`), neither of which populate it
                // either - this function's `MacroblockInfo`s stay
                // comparable via `==` against theirs in
                // `all_three_passes_agree_on_mode_decisions` below, which is
                // about mode decisions specifically, not this diagnostic.
                let (y_block_data, _) = encoder.transform_luma_block(mbx.into(), mby.into(), &info);
                let (u_block_data, v_block_data, _) =
                    encoder.transform_chroma_blocks(mbx.into(), mby.into(), &info);

                if !info.coeffs_skipped {
                    encoder.encode_residual_data(
                        &info,
                        0,
                        mbx.into(),
                        &y_block_data,
                        &u_block_data,
                        &v_block_data,
                    );
                } else {
                    encoder.left_complexity.clear(info.luma_mode != LumaMode::B);
                    encoder.top_complexity[usize::from(mbx)].clear(info.luma_mode != LumaMode::B);
                }

                infos.push(info);
            }
        }

        infos
    }

    /// `(fixture name, width, height, lossy quality, pixel generator)` - just
    /// named to keep `all_three_passes_agree_on_mode_decisions` below under
    /// clippy's `type_complexity` threshold.
    type ModeAgreementFixture = (&'static str, u16, u16, u8, fn(u16, u16) -> Vec<u8>);

    /// The correctness claim that justifies extending `mb_info_cache` to
    /// `encode_image`'s real pass (image-resizer#151): with
    /// `advance_bpred_context` keeping the two dry runs'
    /// `top_b_pred`/`left_b_pred` in step with the real pass, all three
    /// passes must land on the exact same `MacroblockInfo` for every
    /// macroblock. This drives `count_skipped_macroblocks` (pass 1, which
    /// fills `mb_info_cache`) and `collect_token_counts` (pass 2, which
    /// reads it) as `encode_image` does, then separately runs
    /// `independent_real_pass_infos` - a from-scratch, cache-blind
    /// recomputation of what the real pass decides - and asserts all three
    /// results are identical, macroblock for macroblock, for a few fixtures
    /// spanning flat, noisy and mixed-activity content.
    ///
    /// If this ever fails, the fix is not to relax the assertion: it means
    /// the real pass would decide something other than what `encode_image`
    /// now reads from the cache, i.e. the cache extension would silently
    /// change the encoded bitstream.
    #[test]
    fn all_three_passes_agree_on_mode_decisions() {
        let fixtures: [ModeAgreementFixture; 3] = [
            ("flat", 32, 32, 70, flat_rgb),
            ("noisy", 48, 32, 60, noisy_rgb),
            ("mixed", 64, 48, 75, mixed_rgb),
        ];

        for (name, width, height, quality, make_pixels) in fixtures {
            let pixels = make_pixels(width, height);
            let (y_bytes, u_bytes, v_bytes) = convert_image_yuv::<3>(&pixels, width, height);

            let mut encoder = Vp8Encoder::new(Vec::new());
            encoder.setup_encoding(quality, width, height, y_bytes, u_bytes, v_bytes);

            // Pass 1: fills `mb_info_cache`.
            encoder.count_skipped_macroblocks();
            let pass1: Vec<MacroblockInfo> = encoder
                .mb_info_cache
                .iter()
                .map(|info| info.expect("pass 1 fills every entry"))
                .collect();
            encoder.reset_frame_state();

            // Pass 2: reads the same cache (image-resizer#136) - recorded
            // here for completeness, since it is what `encode_image` also
            // reads after this test's pass 3 runs.
            encoder.collect_token_counts();
            let pass2: Vec<MacroblockInfo> = encoder
                .mb_info_cache
                .iter()
                .map(|info| info.expect("pass 1 fills every entry"))
                .collect();
            encoder.reset_frame_state();

            // Pass 3: independently recomputed, deliberately not consulting
            // the cache.
            let pass3 = independent_real_pass_infos(&mut encoder);

            assert_eq!(pass1, pass2, "fixture '{name}': pass 1/2 disagreed");
            assert_eq!(pass1, pass3, "fixture '{name}': pass 1/3 disagreed");
        }
    }

    /// Largest absolute per-element difference between two equal-length
    /// byte buffers - used below to report exactly how far
    /// `apply_loop_filter`'s drift is from zero, rather than just
    /// pass/fail.
    fn max_abs_diff(a: &[u8], b: &[u8]) -> u8 {
        assert_eq!(a.len(), b.len(), "compared buffers have different lengths");
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.abs_diff(y))
            .max()
            .unwrap_or(0)
    }

    /// image-resizer#137 stage 2's core verification. For several fixtures
    /// and qualities (chosen to span `derive_filter_level`'s output from 0
    /// up through a real nonzero level), this:
    ///
    /// 1. runs `Vp8Encoder::encode_image` directly (not through the public
    ///    `WebPEncoder` API) so the test can read `recon_y`/`recon_u`/
    ///    `recon_v` afterwards - the encoder's own simulated *post-filter*
    ///    reconstruction, i.e. what `apply_loop_filter` computed `should`
    ///    be the result of decoding the bitstream just written;
    /// 2. decodes that same raw VP8 bitstream with this crate's own
    ///    `Vp8Decoder` and compares its `Frame::ybuf`/`ubuf`/`vbuf`
    ///    directly against `recon_y`/`recon_u`/`recon_v` - no colour-space
    ///    conversion involved, so any nonzero difference here is purely
    ///    `apply_loop_filter` disagreeing with `Vp8Decoder::loop_filter`;
    /// 3. separately encodes the *same* source pixels through the public
    ///    `WebPEncoder` API (a second, but deterministic and therefore
    ///    byte-identical, encode of the same input) into a real WebP
    ///    container, decodes that with libwebp (`webp::Decoder`, already a
    ///    dev-dependency), and compares its decoded RGB against the same
    ///    `recon_y`/`recon_u`/`recon_v` converted to RGB via `Frame::
    ///    fill_rgb` with `UpsamplingMethod::Bilinear` (`WebPDecoder`'s own
    ///    default, matching libwebp's default "fancy" upsampler).
    ///
    /// Both comparisons are expected to be **exactly** zero: that is the
    /// whole point of mirroring `Vp8Decoder::loop_filter` in
    /// `apply_loop_filter` rather than approximating it - see that
    /// function's doc comment. Every fixture/quality's maximum absolute
    /// difference is printed (`--nocapture`) before the asserts that would
    /// fail on it, so a real regression here reports exact numbers instead
    /// of a bare "assertion failed".
    #[test]
    fn loop_filter_drift() {
        use std::io::Cursor;

        struct Fixture {
            name: &'static str,
            width: u16,
            height: u16,
            pixels: Vec<u8>,
        }

        let fixtures = [
            Fixture {
                name: "flat",
                width: 48,
                height: 32,
                pixels: flat_rgb(48, 32),
            },
            // Non-multiple-of-16 on both axes, so the filter also runs
            // across the partial edge macroblocks the reconstruction
            // planes are sized for.
            Fixture {
                name: "noisy",
                width: 67,
                height: 51,
                pixels: noisy_rgb(67, 51),
            },
            Fixture {
                name: "mixed",
                width: 96,
                height: 64,
                pixels: mixed_rgb(96, 64),
            },
        ];

        // 100 forces `derive_filter_level` to 0 (see
        // `tests/lossy_loop_filter_stage1_byte_identity.rs`'s doc comment)
        // - included as a sanity baseline where `apply_loop_filter` is a
        // no-op - alongside qualities that land on a genuinely nonzero
        // level, which is what actually exercises the filter.
        let qualities = [100u8, 85, 60, 30];

        for fixture in &fixtures {
            for &quality in &qualities {
                let mut encoder = Vp8Encoder::new(Vec::new());
                encoder
                    .encode_image(
                        &fixture.pixels,
                        ColorType::Rgb8,
                        fixture.width,
                        fixture.height,
                        quality,
                    )
                    .unwrap_or_else(|e| {
                        panic!("encode failed for '{}' @ q{quality}: {e}", fixture.name)
                    });

                let vp8_bytes = encoder.writer.clone();
                let recon_y = encoder.recon_y.clone();
                let recon_u = encoder.recon_u.clone();
                let recon_v = encoder.recon_v.clone();
                let filter_level = encoder.frame.filter_level;

                // (2) this crate's own decoder, compared in YUV space -
                // zero colour-space-conversion ambiguity.
                let our_frame = super::super::Vp8Decoder::decode_frame(Cursor::new(vp8_bytes))
                    .unwrap_or_else(|e| {
                        panic!(
                            "this crate's decoder failed for '{}' @ q{quality}: {e}",
                            fixture.name
                        )
                    });
                let our_y_diff = max_abs_diff(&our_frame.ybuf, &recon_y);
                let our_u_diff = max_abs_diff(&our_frame.ubuf, &recon_u);
                let our_v_diff = max_abs_diff(&our_frame.vbuf, &recon_v);

                // (3) libwebp, compared in RGB space via a second,
                // deterministic encode through the public API.
                let mut container = Vec::new();
                let mut public_encoder = crate::WebPEncoder::new(&mut container);
                let params = crate::EncoderParams {
                    use_lossy: true,
                    lossy_quality: quality,
                    ..Default::default()
                };
                public_encoder.set_params(params);
                public_encoder
                    .encode(
                        &fixture.pixels,
                        u32::from(fixture.width),
                        u32::from(fixture.height),
                        crate::ColorType::Rgb8,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "public encode failed for '{}' @ q{quality}: {e}",
                            fixture.name
                        )
                    });
                let libwebp_decoded =
                    webp::Decoder::new(&container).decode().unwrap_or_else(|| {
                        panic!("libwebp failed to decode '{}' @ q{quality}", fixture.name)
                    });

                let recon_frame = Frame {
                    width: fixture.width,
                    height: fixture.height,
                    ybuf: recon_y,
                    ubuf: recon_u,
                    vbuf: recon_v,
                    ..Frame::default()
                };
                let mut recon_rgb =
                    vec![0u8; usize::from(fixture.width) * usize::from(fixture.height) * 3];
                recon_frame.fill_rgb(&mut recon_rgb, crate::UpsamplingMethod::Bilinear);

                assert_eq!(
                    libwebp_decoded.len(),
                    recon_rgb.len(),
                    "fixture '{}' @ q{quality}: libwebp decoded a different-sized buffer \
                     ({} bytes) than the reconstruction-derived RGB buffer ({} bytes)",
                    fixture.name,
                    libwebp_decoded.len(),
                    recon_rgb.len(),
                );
                let libwebp_diff = max_abs_diff(&libwebp_decoded, &recon_rgb);

                eprintln!(
                    "{} @ q{quality} (filter_level={filter_level}): max abs diff - our \
                     decoder Y={our_y_diff} U={our_u_diff} V={our_v_diff}, libwebp RGB={libwebp_diff}",
                    fixture.name,
                );

                assert_eq!(
                    (our_y_diff, our_u_diff, our_v_diff),
                    (0, 0, 0),
                    "fixture '{}' @ q{quality} (filter_level={filter_level}): this crate's \
                     own decoder diverged from the encoder's own post-filter reconstruction \
                     (max abs diff Y={our_y_diff} U={our_u_diff} V={our_v_diff}) - \
                     apply_loop_filter has drifted from Vp8Decoder::loop_filter",
                    fixture.name,
                );
                assert_eq!(
                    libwebp_diff, 0,
                    "fixture '{}' @ q{quality} (filter_level={filter_level}): libwebp \
                     diverged from the encoder's own post-filter reconstruction (max abs \
                     diff, RGB space = {libwebp_diff})",
                    fixture.name,
                );
            }
        }
    }
}
