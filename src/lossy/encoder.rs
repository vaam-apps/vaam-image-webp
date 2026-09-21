use std::io::Write;

use byteorder_lite::{LittleEndian, WriteBytesExt};

use super::arithmetic_encoder::{tree_encode_path, ArithmeticEncoder};
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

/// `trellis_quantize_block` prices rate in real bits (`coeff_token_bit_cost`),
/// not `coeff_rate_estimate`'s cheap proxy, so reusing `mode_decision_lambda`
/// unchanged would under-weight rate by roughly the same factor that made
/// `LAMBDA_SCALE` smaller than the classical bit-domain constant in the first
/// place (see `mode_decision_lambda`'s doc comment). On top of that, trellis
/// measures distortion in the *transform* domain, not the pixel domain
/// `mode_decision_lambda` was calibrated against; per the module doc comment
/// above `trellis_quantize_block`, transform-domain SSE is pixel-domain SSE
/// scaled by the transform's energy gain `c` (`T^T T = c*I`). For this
/// crate's `dct4x4`/`wht4x4`, `c` measures at ~4.0 (checked numerically:
/// impulse coefficients through `idct4x4`, coefficient-domain energy over
/// reconstructed pixel-domain energy comes out to 4.00-4.02 at every
/// position), so a lambda calibrated for real bits against pixel-domain SSE
/// needs dividing by ~4, then multiplying back up by whatever factor
/// `LAMBDA_SCALE` shrank it by relative to the classical real-bit constant.
///
/// `TRELLIS_LAMBDA_MULTIPLIER` is that combined correction, found by
/// sweeping it rather than re-deriving it analytically - the domain-
/// conversion factors above explain its rough size, but `LAMBDA_SCALE`
/// itself is already an empirical fit, not a first-principles constant, so
/// their product is not expected to be exact.
///
/// Two things measured this, not one, because they disagree in an
/// instructive way: at fixed quality (`examples/quick_size`'s byte counts),
/// size keeps dropping well past this value, monotonically, all the way to
/// `TRELLIS_LAMBDA_MULTIPLIER = 50` and beyond - trellis is, correctly,
/// always willing to trade more distortion for fewer bits as lambda grows.
/// But `examples/rd_eval` measures size at *matched DSSIM*, and there the
/// picture is different: below ~3 trellis has essentially no effect (every
/// candidate's real bit-cost saving is too small to outweigh its transform-
/// domain distortion at that lambda, so the search always reproduces plain
/// scalar quantization); by 12 the median ours/libwebp ratio is clearly
/// *worse* than plain scalar (1.49x/1.47x/1.41x vs the 1.26x/1.24x/1.31x
/// baseline) - past some point, the specific coefficients trellis zeroes to
/// save bits (favouring smaller, cheaper-to-signal AC terms) cost DSSIM
/// disproportionately more than the alternative scalar quantization takes
/// to reach the same nominal quality index by other means. 4 is the value
/// in between that measured as a small, consistent win with no regression
/// on any of the three DSSIM targets (1.256x/1.243x/1.295x vs baseline
/// 1.256x/1.244x/1.307x on the six-image Kodak sweep this crate uses for
/// `examples/rd_eval` - see the commit message for the full run). A wider
/// search (finer steps, a per-plane split - e.g. protecting the Y2/DC block
/// with a smaller lambda than the AC blocks, which a quick check found made
/// no measurable difference at this corpus size - or a genuinely different
/// distortion model) might do better; this is the value that was actually
/// verified to help, not a theoretical optimum.
const TRELLIS_LAMBDA_MULTIPLIER: f64 = 4.0;

/// `lambda` for `trellis_quantize_block`: see `TRELLIS_LAMBDA_MULTIPLIER`.
fn trellis_lambda(segment: &Segment) -> f64 {
    mode_decision_lambda(segment) * TRELLIS_LAMBDA_MULTIPLIER
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
/// [`events_from_quantized`]: which `(band, context)` bucket of
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

/// Classifies a quantized coefficient's absolute value into the DCT token
/// that codes it (9.6): `DCT_0` for zero, a literal token for 1..=4, or one
/// of the `DCT_CAT*` category tokens for anything larger. Shared by trellis
/// quantization (`trellis_quantize_block`, which needs this to price a
/// candidate level before choosing it) and event emission
/// (`events_from_quantized`), so the two can never classify the same level
/// differently.
fn dct_token_for_level(abs_level: i32) -> i8 {
    match abs_level {
        0 => DCT_0,
        literal @ 1..=4 => literal as i8,
        5..=6 => DCT_CAT1,
        7..=10 => DCT_CAT2,
        11..=18 => DCT_CAT3,
        19..=34 => DCT_CAT4,
        35..=66 => DCT_CAT5,
        _ => DCT_CAT6,
    }
}

/// Walks already-quantized, zigzag-order coefficients and builds the
/// sequence of token-tree coding events (see `TokenEvent`) for them, plus
/// whether the block has any actual non-zero coefficient (the `has_coeffs`
/// complexity-context bookkeeping threaded between blocks).
///
/// This is the emission half of what used to be a single `tokenize_block`
/// function that also chose `zigzag_block`/`end_of_block_index` by plain
/// scalar division. That quantization decision is now made by
/// `trellis_quantize_block`, which needs this exact walk - context,
/// `skip_eob`, band - to price every candidate level's real bit cost
/// *before* picking one, so quantizing and tokenizing can no longer happen
/// as two independent passes over the same data: whatever
/// `trellis_quantize_block` decides is final, and this only replays it into
/// events (for real bit emission, `Vp8Encoder::emit_residual_events`, or for
/// the token-probability statistics dry run,
/// `Vp8Encoder::collect_token_counts`).
///
/// `has_coeffs` is computed by scanning `zigzag_block` for a non-zero entry,
/// not by comparing `end_of_block_index` to `first_coeff`. Under plain
/// scalar quantization those were always equivalent, because
/// `end_of_block_index` was itself derived from "where's the last non-zero
/// coefficient". Trellis breaks that equivalence: in a probability bucket
/// where an explicit zero token happens to cost fewer bits than the
/// end-of-block token, the search can rationally code one or more explicit
/// zeros before an eventual, still-optimal end-of-block (see
/// `trellis_quantize_block`'s "stop" option), which would make the
/// `end_of_block_index`-based shortcut report `has_coeffs = true` for a
/// block with no non-zero coefficient at all. The decoder's own
/// complexity-context bookkeeping is driven by what it actually decodes, so
/// the encoder has to match that exactly - a mismatch here would desync the
/// context used for later blocks from what the decoder computes, while
/// still producing a technically-decodable (but wrongly-decoded-context,
/// silently-corrupt) bitstream.
fn events_from_quantized(
    zigzag_block: &[i32; 16],
    end_of_block_index: usize,
    plane: Plane,
    initial_context: usize,
) -> (Vec<TokenEvent>, bool) {
    let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };

    assert!(initial_context <= 2);
    let mut complexity = initial_context;

    let mut events = Vec::new();
    let mut skip_eob = false;
    let mut has_coeffs = false;

    for index in first_coeff..end_of_block_index {
        let coeff = zigzag_block[index];
        if coeff != 0 {
            has_coeffs = true;
        }

        let band = usize::from(COEFF_BANDS[index]);
        let start_index = if skip_eob { 2 } else { 0 };

        let token = dct_token_for_level(coeff.abs());
        // never going to have an end of block right after a 0, so skip
        // checking next coeff's start index
        skip_eob = token == DCT_0;

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

    (events, has_coeffs)
}

/// Per-`(plane, band, context)` bucket branch counts for every internal node
/// of the coefficient token tree (`DCT_TOKEN_TREE`): `[false_count,
/// true_count]` for `token_probs[plane][band][context][node]`. Same shape as
/// `TokenProbTables`, but counting how often each branch was actually taken
/// this frame instead of holding a probability. Built by
/// `Vp8Encoder::collect_token_counts`, consumed by
/// `derive_updated_token_probs`.
type TokenCounts = [[[[[u64; 2]; NUM_DCT_TOKENS - 1]; 3]; 8]; 4];

/// Adds one macroblock's worth of already-tagged token events (see
/// `TokenEvent`, and the `(Plane, TokenEvent)` shape `transform_luma_block`
/// / `transform_luma_blocks_4x4` / `transform_chroma_blocks` return) into
/// `counts`, by replaying the exact same root-to-leaf tree walk
/// `write_with_tree_start_index` performs when actually writing a token
/// (`tree_encode_path`) and counting each branch instead of encoding it.
fn accumulate_tagged_events(counts: &mut TokenCounts, events: &[(Plane, TokenEvent)]) {
    for (plane, event) in events {
        for (bit, prob_index) in tree_encode_path(&DCT_TOKEN_TREE, event.token, event.start_index)
        {
            counts[*plane as usize][event.band][event.context][prob_index][usize::from(bit)] += 1;
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
    false_count as f64 * branch_bit_cost(prob, false) + true_count as f64 * branch_bit_cost(prob, true)
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

// --- Trellis (rate-distortion optimised) quantization -----------------------
//
// Scalar quantization (the old `tokenize_block`'s `block[i] / quant`) rounds
// each coefficient independently, toward zero, with no regard for what it
// costs to *encode* that choice. Trellis quantization instead searches, per
// block, over which level to send at every zigzag position - including the
// position at which to stop the block early (end-of-block) - to minimise a
// Lagrangian `distortion + lambda * rate` over the whole block at once. This
// is genuinely a dynamic program over `(position, context)`, not a
// per-coefficient threshold: a coefficient's token cost depends on the
// *previous* token's magnitude class (0, 1, or >1 - `TokenEvent::context`)
// and the current band, so the context a later position is priced under
// depends on what was chosen at every earlier position, and the DP has to
// account for that when deciding what's optimal now. `trellis_quantize_
// block` below is that DP; `CandidateSet` is the per-position candidate
// levels it chooses among.
//
// Distortion is measured as squared error between the dequantized candidate
// and the *original* transform coefficient, not in the pixel domain. This is
// legitimate because `transform::dct4x4`/`wht4x4` (like any DCT/Hadamard
// transform) are orthogonal up to a fixed scale factor: writing the forward
// transform as a matrix `T`, `T^T T = c * I` for some constant `c` shared by
// every coefficient. For an error `deltaX` introduced in the coefficient
// domain, the resulting pixel-domain error after `idct4x4`/`iwht4x4` is
// `T^-1 deltaX = (T^T / c) deltaX`, whose squared norm is `(1/c) *
// ||deltaX||^2` - i.e. transform-domain SSE is pixel-domain SSE scaled by a
// single constant, so ranking candidates by transform-domain SSE ranks them
// identically to ranking by pixel-domain SSE. `lambda` is `mode_decision_
// lambda`'s value (so this search and mode decision are choosing points on
// the same underlying rate/distortion curve for the same image) times
// `TRELLIS_LAMBDA_MULTIPLIER`, a correction for the two ways this search's
// units differ from what `mode_decision_lambda` was calibrated for - real
// bits instead of `coeff_rate_estimate`'s cheap proxy, and transform-domain
// instead of pixel-domain distortion; see `trellis_lambda` and
// `TRELLIS_LAMBDA_MULTIPLIER`'s doc comment for both, and the measurements
// behind the constant's actual value.

/// The candidate levels trellis considers at one zigzag position: the plain
/// scalar-quantised level (truncating division, same as the old scalar-only
/// path), that level moved one step toward zero, and zero - deduplicated.
/// This is the minimum candidate set the search is required to use; a wider
/// set (e.g. also trying one step *away* from zero, toward round-to-nearest)
/// could find marginally better roundings in some blocks but isn't
/// implemented here. A fixed-size array rather than a `Vec`, since this is
/// built for every position of every block trellis-quantizes and a heap
/// allocation there would be wasteful.
#[derive(Clone, Copy)]
struct CandidateSet {
    levels: [i32; 3],
    count: usize,
}

impl CandidateSet {
    const EMPTY: CandidateSet = CandidateSet {
        levels: [0; 3],
        count: 0,
    };

    fn new(scalar_level: i32) -> Self {
        let toward_zero = scalar_level - scalar_level.signum();
        let mut set = CandidateSet::EMPTY;
        for &level in &[scalar_level, toward_zero, 0] {
            if !set.levels[..set.count].contains(&level) {
                set.levels[set.count] = level;
                set.count += 1;
            }
        }
        set
    }

    fn as_slice(&self) -> &[i32] {
        &self.levels[..self.count]
    }
}

/// Real entropy cost, in bits, of one candidate token - the tree-traversal
/// cost (`tree_encode_path`, the same walk `write_with_tree_start_index`
/// performs when actually writing a token, priced via `branch_bit_cost`)
/// plus, for a category token, its "extra" magnitude bits and the sign bit.
/// Both of those are coded with fixed, non-adapted probabilities
/// (`PROB_DCT_CAT`, and a flat `128` for the sign - matching `write_flag`),
/// exactly mirroring what `emit_residual_events` actually writes for a
/// nonzero token. Used by `trellis_quantize_block` to price a candidate
/// level without writing it.
fn coeff_token_bit_cost(
    token_probs: &[Prob; NUM_DCT_TOKENS - 1],
    token: i8,
    start_index: usize,
    quantized_value: i32,
) -> f64 {
    let mut bits = 0.0;
    for (bit, prob_index) in tree_encode_path(&DCT_TOKEN_TREE, token, start_index) {
        bits += branch_bit_cost(token_probs[prob_index], bit);
    }

    if token == DCT_EOB || token == DCT_0 {
        return bits;
    }

    if token >= DCT_CAT1 {
        let category = token;
        let category_probs = PROB_DCT_CAT[(category - DCT_CAT1) as usize];
        let value = quantized_value.abs();
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
            bits += branch_bit_cost(prob, extra_bool);
            mask >>= 1;
        }
    }

    // sign bit: fixed probability 128, matching `write_flag`.
    bits += branch_bit_cost(128, quantized_value.is_negative());

    bits
}

/// A trellis DP decision at one `(position, context)` state: either place
/// the end-of-block here (nothing at this position or later is coded), or
/// code `level` here and continue to the next position.
#[derive(Clone, Copy)]
enum TrellisChoice {
    Stop,
    Level(i32),
}

/// Rate-distortion optimised ("trellis") quantization of one block, in
/// place of plain scalar rounding. See the module doc comment above for the
/// Lagrangian, the candidate set, and the orthogonality argument for scoring
/// distortion in the transform domain.
///
/// `natural_block` holds the raw (unquantized) transform coefficients in
/// natural (raster) order, exactly as `transform::dct4x4`/`wht4x4` produce
/// them - the same input `tokenize_block` used to take. `initial_context` is
/// the `(0/1/2)` context this block's *first* coded position enters with,
/// i.e. `left_complexity + top_complexity` from the neighbouring blocks,
/// same convention as the old scalar path.
///
/// Returns the chosen coefficients in natural order, ready to dequantize and
/// inverse-transform for reconstruction (multiply by `dc_quant`/`ac_quant`
/// and run `idct4x4`/`iwht4x4`, exactly like the old scalar path did); the
/// token-coding events for those coefficients (see `TokenEvent`), built in
/// the same pass since the winning path's bit costs are already known - no
/// separate re-tokenization is needed, and (per `events_from_quantized`'s
/// doc comment) re-deriving `has_coeffs` independently would risk getting it
/// wrong; and whether the block has any actual non-zero coefficient.
fn trellis_quantize_block(
    natural_block: &[i32; 16],
    plane: Plane,
    initial_context: usize,
    dc_quant: i16,
    ac_quant: i16,
    token_probs_plane: &[[[Prob; NUM_DCT_TOKENS - 1]; 3]; 8],
    lambda: f64,
) -> ([i32; 16], Vec<TokenEvent>, bool) {
    let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
    assert!(initial_context <= 2);

    // Zigzag-order the raw coefficients: the token tree's band/context
    // structure (and so its real bit cost) is defined over zigzag
    // (frequency) order, not the DCT's natural raster order.
    let mut raw = [0i64; 16];
    for i in first_coeff..16 {
        raw[i] = i64::from(natural_block[usize::from(ZIGZAG[i])]);
    }

    let quant_step = |i: usize| -> i64 {
        if i == 0 {
            i64::from(dc_quant)
        } else {
            i64::from(ac_quant)
        }
    };

    // Suffix sum of squared raw coefficients: the transform-domain
    // distortion of implicitly zeroing every position from `i` to 15 - what
    // ending the block at `i` (an explicit end-of-block, or simply running
    // off the end at 16) costs, since a dequantized zero coefficient is
    // exactly 0.
    let mut suffix_distortion = [0i64; 17];
    for i in (first_coeff..16).rev() {
        suffix_distortion[i] = suffix_distortion[i + 1] + raw[i] * raw[i];
    }

    let candidates: [CandidateSet; 16] = std::array::from_fn(|i| {
        if i < first_coeff {
            CandidateSet::EMPTY
        } else {
            let scalar_level = (raw[i] / quant_step(i)) as i32;
            CandidateSet::new(scalar_level)
        }
    });

    // Backward DP: `dp[i][ctx]` is the minimal `distortion + lambda * rate`
    // of positions `i..16`, given position `i` is entered with token-tree
    // context `ctx` - the previous token's magnitude class. `dp[16][_] =
    // 0.0`: reaching position 16 needs no further coding at all (matches
    // `events_from_quantized`'s `end_of_block_index < 16` check - the
    // decoder infers the rest of the block is zero without any token).
    //
    // The cost of choosing a candidate at position `i` includes `dp[i +
    // 1][context after that candidate]`, so what's optimal here depends on
    // what it makes possible afterwards, and vice versa - a real dynamic
    // program over `(position, context)`, not sixteen independent
    // thresholding decisions.
    let mut dp = [[0.0f64; 3]; 17];
    let mut choice = [[TrellisChoice::Stop; 3]; 17];

    for i in (first_coeff + 1..16).rev() {
        let band = usize::from(COEFF_BANDS[i]);
        for ctx in 0..3usize {
            // After a zero token, `write_with_tree_start_index` starts the
            // *next* token's tree walk at index 2 (`skip_eob`), which skips
            // the branch that would encode "end of block here" - the
            // format simply disallows placing an explicit end-of-block
            // right after an explicit zero (it would always have been one
            // token cheaper to just stop at the zero's position instead).
            // So "stop here" is only a legal choice when `ctx != 0`.
            let start_index = if ctx == 0 { 2 } else { 0 };
            let probs = &token_probs_plane[band][ctx];

            let mut best_cost = f64::INFINITY;
            let mut best_choice = TrellisChoice::Stop;

            if start_index == 0 {
                let eob_bits = coeff_token_bit_cost(probs, DCT_EOB, start_index, 0);
                best_cost = lambda * eob_bits + suffix_distortion[i] as f64;
                best_choice = TrellisChoice::Stop;
            }

            for &level in candidates[i].as_slice() {
                let token = dct_token_for_level(level.abs());
                let bits = coeff_token_bit_cost(probs, token, start_index, level);
                let diff = i64::from(level) * quant_step(i) - raw[i];
                let distortion = (diff * diff) as f64;
                let next_ctx = match token {
                    DCT_0 => 0,
                    DCT_1 => 1,
                    _ => 2,
                };
                let cost = distortion + lambda * bits + dp[i + 1][next_ctx];
                if cost < best_cost {
                    best_cost = cost;
                    best_choice = TrellisChoice::Level(level);
                }
            }

            dp[i][ctx] = best_cost;
            choice[i][ctx] = best_choice;
        }
    }

    // The block's first coded position is special: a block's first token
    // always starts its tree walk at index 0, regardless of
    // `initial_context` - that context comes from neighbouring blocks' last
    // token, not from anything coded so far *in this block* - matching the
    // old scalar path's `skip_eob = false` initial value. So this can't
    // reuse the `ctx == 0 => start_index = 2` rule the backward pass above
    // uses for every later position.
    let band0 = usize::from(COEFF_BANDS[first_coeff]);
    let probs0 = &token_probs_plane[band0][initial_context];

    let mut best_cost = lambda * coeff_token_bit_cost(probs0, DCT_EOB, 0, 0)
        + suffix_distortion[first_coeff] as f64;
    let mut best_choice = TrellisChoice::Stop;

    for &level in candidates[first_coeff].as_slice() {
        let token = dct_token_for_level(level.abs());
        let bits = coeff_token_bit_cost(probs0, token, 0, level);
        let diff = i64::from(level) * quant_step(first_coeff) - raw[first_coeff];
        let distortion = (diff * diff) as f64;
        let next_ctx = match token {
            DCT_0 => 0,
            DCT_1 => 1,
            _ => 2,
        };
        let cost = distortion + lambda * bits + dp[first_coeff + 1][next_ctx];
        if cost < best_cost {
            best_cost = cost;
            best_choice = TrellisChoice::Level(level);
        }
    }

    // Forward backtrack: replay the winning choices to build the zigzag
    // coefficient array and find where the block actually ends.
    let mut zigzag = [0i32; 16];
    let end_of_block_index = match best_choice {
        TrellisChoice::Stop => first_coeff,
        TrellisChoice::Level(level) => {
            zigzag[first_coeff] = level;
            let mut ctx = match dct_token_for_level(level.abs()) {
                DCT_0 => 0,
                DCT_1 => 1,
                _ => 2,
            };
            let mut i = first_coeff + 1;
            loop {
                if i == 16 {
                    break 16;
                }
                match choice[i][ctx] {
                    TrellisChoice::Stop => break i,
                    TrellisChoice::Level(level) => {
                        zigzag[i] = level;
                        ctx = match dct_token_for_level(level.abs()) {
                            DCT_0 => 0,
                            DCT_1 => 1,
                            _ => 2,
                        };
                        i += 1;
                    }
                }
            }
        }
    };

    let (events, has_coeffs) =
        events_from_quantized(&zigzag, end_of_block_index, plane, initial_context);

    let mut natural = [0i32; 16];
    for i in first_coeff..16 {
        natural[usize::from(ZIGZAG[i])] = zigzag[i];
    }

    (natural, events, has_coeffs)
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

    // 13 in specification, matches read_residual_data in the decoder.
    //
    // Unlike the old `encode_residual_data`, this does no quantization or
    // tokenization itself - it only writes bits for events already produced
    // by `trellis_quantize_block` (inside `transform_luma_block` /
    // `transform_luma_blocks_4x4` / `transform_chroma_blocks`, which run
    // before this is called). Trellis has to know a token's real bit cost
    // *before* choosing it, so the quantizing decision and the tokenizing
    // are made together, well before the bits are actually written; this
    // only replays that decision. `events` must be in encode order (Y2,
    // then the 16 luma sub-blocks, then all of U, then all of V) - exactly
    // what those three methods build them in.
    fn emit_residual_events(&mut self, partition_index: usize, events: &[(Plane, TokenEvent)]) {
        // `self.token_probs` is `Copy`; taking a local copy here avoids
        // borrowing `self` both immutably (for the probabilities) and
        // mutably (for `self.partitions[partition_index]`) at once.
        let token_probs = self.token_probs;
        let encoder = &mut self.partitions[partition_index];

        for (plane, event) in events {
            let probs = &token_probs[*plane as usize][event.band][event.context];
            encoder.write_with_tree_start_index(
                &DCT_TOKEN_TREE,
                probs,
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
        // `token_probs` from them now. `emit_residual_events` reads
        // `self.token_probs` directly, so setting it here is what makes the
        // real pass below use exactly what the header just advertised - see
        // `derive_updated_token_probs` for the update decision itself. It's
        // also what `trellis_quantize_block` prices candidates against in
        // the real pass below, though not in this dry run or the skip-rate
        // one above - both of those still run before `token_probs` is
        // known, so their trellis decisions are made (and then thrown away)
        // against the `COEFF_PROBS` default. That's a known, small
        // bootstrapping approximation: this dry run's job is to *learn*
        // `token_probs`, so it can't yet know the value it's computing.
        let token_counts = self.collect_token_counts();
        self.reset_frame_state();
        self.token_probs = derive_updated_token_probs(&token_counts);

        self.encode_compressed_frame_header();

        // encode residual partitions first
        let mut events: Vec<(Plane, TokenEvent)> = Vec::new();
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
                //
                // These two also run trellis quantization (`trellis_
                // quantize_block`) and update `self.left_complexity` /
                // `self.top_complexity` themselves as they go, since with
                // trellis the quantized levels depend on that running
                // context - unlike the old scalar-only path, it can no
                // longer be recomputed independently afterwards. Whenever a
                // macroblock's mode-decision trial found every coefficient
                // to be zero (`coeffs_skipped`), trellis - whose candidate
                // set at a position whose scalar level is 0 is just `{0}` -
                // is forced to reproduce that same all-zero result here, so
                // there's nothing left to separately "clear".
                events.clear();
                self.transform_luma_block(mbx.into(), mby.into(), &macroblock_info, &mut events);
                self.transform_chroma_blocks(
                    mbx.into(),
                    mby.into(),
                    macroblock_info.chroma_mode,
                    &mut events,
                );

                if !macroblock_info.coeffs_skipped {
                    self.emit_residual_events(partition_index, &events);
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
        let mut events: Vec<(Plane, TokenEvent)> = Vec::new();

        for mby in 0..self.macroblock_height {
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];
            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let info = self.choose_macroblock_info(mbx.into(), mby.into());

                // `transform_luma_block` / `transform_chroma_blocks` also
                // run trellis quantization and thread real complexity
                // context now (see `encode_image`'s equivalent comment);
                // `events` is discarded here, same as the old scalar-only
                // path discarded its return values - this dry run only
                // needs `info.coeffs_skipped`, which mode decision already
                // determined.
                events.clear();
                self.transform_luma_block(mbx.into(), mby.into(), &info, &mut events);
                self.transform_chroma_blocks(mbx.into(), mby.into(), info.chroma_mode, &mut events);

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
    /// Same two-pass shape as `count_skipped_macroblocks` just above, and for
    /// the same reason: mode decision only depends on pixel data and border
    /// state, never on entropy-coding probabilities (see that method's doc
    /// comment), so this can run mode decision for real and accumulate every
    /// block's actual trellis-chosen tokens (`accumulate_tagged_events`).
    /// Trellis quantization itself *does* depend on entropy-coding
    /// probabilities (that's the whole point), but by the time this runs
    /// `self.token_probs` is still the `COEFF_PROBS` default - see
    /// `encode_image`'s comment on that bootstrapping approximation. The
    /// border/b_pred/complexity state this mutates is fully reset by
    /// `reset_frame_state` immediately afterwards, same as after
    /// `count_skipped_macroblocks`.
    fn collect_token_counts(&mut self) -> TokenCounts {
        let mut counts: TokenCounts = [[[[[0u64; 2]; NUM_DCT_TOKENS - 1]; 3]; 8]; 4];
        let mut events: Vec<(Plane, TokenEvent)> = Vec::new();

        for mby in 0..self.macroblock_height {
            self.left_complexity = Complexity::default();
            self.left_b_pred = [IntraMode::default(); 4];
            self.left_border_y = [129u8; 16 + 1];
            self.left_border_u = [129u8; 8 + 1];
            self.left_border_v = [129u8; 8 + 1];

            for mbx in 0..self.macroblock_width {
                let mbx = usize::from(mbx);
                let mby = usize::from(mby);
                let info = self.choose_macroblock_info(mbx, mby);

                events.clear();
                self.transform_luma_block(mbx, mby, &info, &mut events);
                self.transform_chroma_blocks(mbx, mby, info.chroma_mode, &mut events);

                if info.coeffs_skipped {
                    // matches `encode_image`'s handling of a skipped
                    // macroblock: no residual data (and so no tokens) is
                    // ever coded for it. The complexity context it leaves
                    // for its neighbours is already all-zero, via the same
                    // reasoning as `count_skipped_macroblocks` - no separate
                    // clearing needed.
                    continue;
                }

                accumulate_tagged_events(&mut counts, &events);
            }
        }

        counts
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
    /// initial, "no macroblocks encoded yet" values. Called from
    /// `setup_encoding` (first use) and from `encode_image` after each of
    /// its two dry runs - `count_skipped_macroblocks` and
    /// `collect_token_counts` - both of which mutate all of this exactly
    /// like the real pass would, and must not leak into it or into each
    /// other.
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
    // 4. Trellis-quantizes the block (`trellis_quantize_block`) and
    //    dequantizes each subblock
    // 5. Reconstructs from the dequantized coefficients - this populates the
    //    borders for the next macroblock, and must use the exact same
    //    coefficients tokenized into `events` (see `trellis_quantize_
    //    block`'s doc comment): reconstructing from anything else would
    //    desync this encoder's own predictions from what the decoder will
    //    reconstruct from the bits actually written.
    //
    // Pushes this macroblock's Y2 and luma token events onto `events`, in
    // the same Y2-then-16-subblocks order `emit_residual_events` /
    // `accumulate_tagged_events` expect, and updates `self.left_complexity`
    // / `self.top_complexity` as it goes (unlike the old scalar-only path,
    // where that threading happened later, in a separate re-tokenization
    // pass - trellis needs the real running context available *before* it
    // can quantize, not after).
    fn transform_luma_block(
        &mut self,
        mbx: usize,
        mby: usize,
        macroblock_info: &MacroblockInfo,
        events: &mut Vec<(Plane, TokenEvent)>,
    ) {
        if macroblock_info.luma_mode == LumaMode::B {
            if let Some(bpred_modes) = macroblock_info.luma_bpred {
                self.transform_luma_blocks_4x4(bpred_modes, mbx, mby, events);
            } else {
                panic!("Invalid, need bpred modes for luma mode B");
            }
            return;
        }

        let mut y_with_border =
            self.get_predicted_luma_block_16x16(macroblock_info.luma_mode, mbx, mby);
        let luma_blocks = self.get_luma_blocks_from_predicted_16x16(&y_with_border, mbx, mby);

        let segment = self.segments[macroblock_info.segment_id.unwrap_or(0)];
        let lambda = trellis_lambda(&segment);

        // Y2: the WHT of each of the 16 luma blocks' own DC coefficient
        // (13.2/14.3 in the spec) - trellis-quantized first, since its
        // dequantized/inverse-WHT output supplies the DC term every luma
        // block below reconstructs from. B_PRED has no Y2 block; see
        // `transform_luma_blocks_4x4`.
        let mut y2 = get_coeffs0_from_block(&luma_blocks);
        transform::wht4x4(&mut y2);

        let y2_context = self.left_complexity.y2 + self.top_complexity[mbx].y2;
        let (mut y2_dequant, y2_events, y2_has_coeffs) = trellis_quantize_block(
            &y2,
            Plane::Y2,
            y2_context.into(),
            segment.y2dc,
            segment.y2ac,
            &self.token_probs[Plane::Y2 as usize],
            lambda,
        );
        events.extend(y2_events.into_iter().map(|event| (Plane::Y2, event)));
        self.left_complexity.y2 = if y2_has_coeffs { 1 } else { 0 };
        self.top_complexity[mbx].y2 = if y2_has_coeffs { 1 } else { 0 };

        for (k, coeff) in y2_dequant.iter_mut().enumerate() {
            let quant = if k > 0 { segment.y2ac } else { segment.y2dc };
            *coeff *= i32::from(quant);
        }
        transform::iwht4x4(&mut y2_dequant);

        // Now trellis-quantize each of the 16 luma AC blocks - their own DC
        // (`Plane::YCoeff1`'s `first_coeff == 1`) is skipped, since Y2
        // already carries it - substitute the inverse-WHT'd Y2 term back in
        // as each block's DC, and IDCT/reconstruct exactly like the
        // decoder's own residual reconstruction.
        for y in 0usize..4 {
            let mut left = self.left_complexity.y[y];
            for x in 0..4 {
                let i = y * 4 + x;
                let block: &[i32; 16] = luma_blocks[i * 16..][..16].try_into().unwrap();

                let top = self.top_complexity[mbx].y[x];
                let context = left + top;

                let (mut natural, block_events, has_coeffs) = trellis_quantize_block(
                    block,
                    Plane::YCoeff1,
                    context.into(),
                    segment.ydc,
                    segment.yac,
                    &self.token_probs[Plane::YCoeff1 as usize],
                    lambda,
                );
                events.extend(block_events.into_iter().map(|event| (Plane::YCoeff1, event)));

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[x] = if has_coeffs { 1 } else { 0 };

                for v in natural[1..].iter_mut() {
                    *v *= i32::from(segment.yac);
                }
                natural[0] = y2_dequant[i];
                transform::idct4x4(&mut natural);

                let y0 = 1 + y * 4;
                let x0 = 1 + x * 4;
                add_residue(&mut y_with_border, &natural, y0, x0, LUMA_STRIDE);
            }
            self.left_complexity.y[y] = left;
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * LUMA_STRIDE + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * LUMA_STRIDE + x + 1];
        }
    }

    // this is for transforming the luma blocks for each subblock independently
    // meaning the luma mode is B
    //
    // Unlike the 16x16 modes, B_PRED's sub-blocks are not independent: each
    // one predicts from the *reconstructed* pixels of earlier sub-blocks in
    // the same macroblock (`predict_b*pred` reads the border buffer this
    // loop just wrote into). So trellis-quantizing and reconstructing a
    // sub-block has to happen before the next sub-block is predicted, same
    // requirement `trial_luma_bpred`'s doc comment explains for the RD
    // search - this is the real-encode counterpart of that.
    fn transform_luma_blocks_4x4(
        &mut self,
        bpred_modes: [IntraMode; 16],
        mbx: usize,
        mby: usize,
        events: &mut Vec<(Plane, TokenEvent)>,
    ) {
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
        let lambda = trellis_lambda(&segment);

        for sby in 0usize..4 {
            let mut left = self.left_complexity.y[sby];
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

                let top = self.top_complexity[mbx].y[sbx];
                let context = left + top;

                let (mut natural, block_events, has_coeffs) = trellis_quantize_block(
                    &current_subblock,
                    Plane::YCoeff0,
                    context.into(),
                    segment.ydc,
                    segment.yac,
                    &self.token_probs[Plane::YCoeff0 as usize],
                    lambda,
                );
                events.extend(block_events.into_iter().map(|event| (Plane::YCoeff0, event)));

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].y[sbx] = if has_coeffs { 1 } else { 0 };

                for (index, v) in natural.iter_mut().enumerate() {
                    let quant = if index > 0 { segment.yac } else { segment.ydc };
                    *v *= i32::from(quant);
                }
                transform::idct4x4(&mut natural);
                add_residue(&mut y_with_border, &natural, y0, x0, stride);
            }
            self.left_complexity.y[sby] = left;
        }

        // set borders from values
        for (y, border_value) in self.left_border_y.iter_mut().enumerate() {
            *border_value = y_with_border[y * stride + 16];
        }

        for (x, border_value) in self.top_border_y[mbx * 16..][..16].iter_mut().enumerate() {
            *border_value = y_with_border[16 * stride + x + 1];
        }
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

    /// Trellis-quantizes and reconstructs both chroma planes, pushing their
    /// token events onto `events` - all of U's four sub-blocks, in raster
    /// order, followed by all of V's (13.3 in the spec: chroma residual data
    /// is U-plane-then-V-plane, not interleaved - `emit_residual_events` /
    /// `accumulate_tagged_events` need `events` in exactly that order).
    /// Updates `self.left_complexity`/`self.top_complexity`'s `u`/`v` fields
    /// as it goes, same reasoning as `transform_luma_block`.
    fn transform_chroma_blocks(
        &mut self,
        mbx: usize,
        mby: usize,
        chroma_mode: ChromaMode,
        events: &mut Vec<(Plane, TokenEvent)>,
    ) {
        let stride = CHROMA_STRIDE;
        let segment = self.segments[0];
        let lambda = trellis_lambda(&segment);

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

        for y in 0usize..2 {
            let mut left = self.left_complexity.u[y];
            for x in 0usize..2 {
                let i = y * 2 + x;
                let block: &[i32; 16] = u_blocks[i * 16..][..16].try_into().unwrap();

                let top = self.top_complexity[mbx].u[x];
                let context = left + top;

                let (mut natural, block_events, has_coeffs) = trellis_quantize_block(
                    block,
                    Plane::Chroma,
                    context.into(),
                    segment.uvdc,
                    segment.uvac,
                    &self.token_probs[Plane::Chroma as usize],
                    lambda,
                );
                events.extend(block_events.into_iter().map(|event| (Plane::Chroma, event)));

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].u[x] = if has_coeffs { 1 } else { 0 };

                for (index, v) in natural.iter_mut().enumerate() {
                    let quant = if index > 0 { segment.uvac } else { segment.uvdc };
                    *v *= i32::from(quant);
                }
                transform::idct4x4(&mut natural);
                add_residue(&mut predicted_u, &natural, 1 + y * 4, 1 + x * 4, stride);
            }
            self.left_complexity.u[y] = left;
        }

        for y in 0usize..2 {
            let mut left = self.left_complexity.v[y];
            for x in 0usize..2 {
                let i = y * 2 + x;
                let block: &[i32; 16] = v_blocks[i * 16..][..16].try_into().unwrap();

                let top = self.top_complexity[mbx].v[x];
                let context = left + top;

                let (mut natural, block_events, has_coeffs) = trellis_quantize_block(
                    block,
                    Plane::Chroma,
                    context.into(),
                    segment.uvdc,
                    segment.uvac,
                    &self.token_probs[Plane::Chroma as usize],
                    lambda,
                );
                events.extend(block_events.into_iter().map(|event| (Plane::Chroma, event)));

                left = if has_coeffs { 1 } else { 0 };
                self.top_complexity[mbx].v[x] = if has_coeffs { 1 } else { 0 };

                for (index, v) in natural.iter_mut().enumerate() {
                    let quant = if index > 0 { segment.uvac } else { segment.uvdc };
                    *v *= i32::from(quant);
                }
                transform::idct4x4(&mut natural);
                add_residue(&mut predicted_v, &natural, 1 + y * 4, 1 + x * 4, stride);
            }
            self.left_complexity.v[y] = left;
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
mod trellis_tests {
    use super::*;
    use rand::Rng;

    /// Reference scalar quantization: plain truncating division in zigzag
    /// order, and "one past the last non-zero position" as end-of-block -
    /// exactly what the old, pre-trellis `tokenize_block` did.
    fn scalar_quantize_zigzag(
        natural_block: &[i32; 16],
        plane: Plane,
        dc_quant: i16,
        ac_quant: i16,
    ) -> [i32; 16] {
        let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
        let mut zigzag = [0i32; 16];
        for i in first_coeff..16 {
            let zigzag_index = usize::from(ZIGZAG[i]);
            let quant = if zigzag_index > 0 { ac_quant } else { dc_quant };
            zigzag[i] = natural_block[zigzag_index] / i32::from(quant);
        }
        zigzag
    }

    /// The same `distortion + lambda * rate` total `trellis_quantize_
    /// block`'s DP computes internally, re-derived from the outside: for a
    /// full-length (positions `0..16`, already 0 past wherever coding
    /// actually stopped) zigzag array and the token events that were
    /// (or would be) emitted for it. Used to check the DP's own optimality
    /// claim against independently-computed reference numbers, without
    /// needing to know where the block's end-of-block position was.
    #[allow(clippy::too_many_arguments)]
    fn coding_cost(
        raw: &[i64; 16],
        zigzag: &[i32; 16],
        events: &[TokenEvent],
        first_coeff: usize,
        dc_quant: i16,
        ac_quant: i16,
        probs: &[[[Prob; NUM_DCT_TOKENS - 1]; 3]; 8],
        lambda: f64,
    ) -> f64 {
        let mut cost = 0.0;
        for i in first_coeff..16 {
            let step = if i == 0 { dc_quant } else { ac_quant };
            let diff = i64::from(zigzag[i]) * i64::from(step) - raw[i];
            cost += (diff * diff) as f64;
        }
        for event in events {
            let bits = coeff_token_bit_cost(
                &probs[event.band][event.context],
                event.token,
                event.start_index,
                event.quantized_value,
            );
            cost += lambda * bits;
        }
        cost
    }

    /// Trellis quantization is a search over a superset of what plain scalar
    /// quantization does (scalar's own level is always one of the
    /// candidates, and "stop now" is always at least as available as
    /// scalar's implicit natural end-of-block), so for *any* block and
    /// *any* lambda, its `distortion + lambda * rate` total can never be
    /// worse than scalar's. This is the DP's core correctness property,
    /// independent of whether a given lambda happens to help real-world
    /// compression - that's `examples/rd_eval`'s job, this is "did the
    /// search actually search".
    #[test]
    fn trellis_cost_never_exceeds_scalar() {
        let mut rng = rand::thread_rng();
        for plane in [Plane::YCoeff1, Plane::Y2, Plane::Chroma, Plane::YCoeff0] {
            let first_coeff = if plane == Plane::YCoeff1 { 1 } else { 0 };
            for _ in 0..2000 {
                let mut natural = [0i32; 16];
                for v in natural.iter_mut() {
                    *v = rng.gen_range(-600..=600);
                }
                let dc_quant = rng.gen_range(4..=157);
                let ac_quant = rng.gen_range(4..=284);
                let initial_context = rng.gen_range(0..=2usize);
                // Sweep several orders of magnitude, since the DP's branch
                // structure (which candidate/stop option wins) changes
                // qualitatively across that range.
                let lambda = 10f64.powf(rng.gen_range(-3.0..4.0));
                let probs = &COEFF_PROBS[plane as usize];

                let mut raw = [0i64; 16];
                for i in first_coeff..16 {
                    raw[i] = i64::from(natural[usize::from(ZIGZAG[i])]);
                }

                let scalar_zigzag = scalar_quantize_zigzag(&natural, plane, dc_quant, ac_quant);
                let scalar_eob = scalar_zigzag
                    .iter()
                    .rev()
                    .position(|x| *x != 0)
                    .map_or(0, |last| 16 - last);
                let (scalar_events, _) =
                    events_from_quantized(&scalar_zigzag, scalar_eob, plane, initial_context);
                let scalar_cost = coding_cost(
                    &raw,
                    &scalar_zigzag,
                    &scalar_events,
                    first_coeff,
                    dc_quant,
                    ac_quant,
                    probs,
                    lambda,
                );

                let (trellis_natural, trellis_events, _) = trellis_quantize_block(
                    &natural,
                    plane,
                    initial_context,
                    dc_quant,
                    ac_quant,
                    probs,
                    lambda,
                );
                let mut trellis_zigzag = [0i32; 16];
                for i in first_coeff..16 {
                    trellis_zigzag[i] = trellis_natural[usize::from(ZIGZAG[i])];
                }
                let trellis_cost = coding_cost(
                    &raw,
                    &trellis_zigzag,
                    &trellis_events,
                    first_coeff,
                    dc_quant,
                    ac_quant,
                    probs,
                    lambda,
                );

                assert!(
                    trellis_cost <= scalar_cost + 1e-6,
                    "trellis cost {trellis_cost} exceeds scalar cost {scalar_cost} \
                     (plane {:?}, natural {:?}, dc_quant {dc_quant}, ac_quant {ac_quant}, \
                     initial_context {initial_context}, lambda {lambda})",
                    plane as usize,
                    natural,
                );
            }
        }
    }

    /// The decoder's complexity context for a block is driven by whether it
    /// actually decodes a non-zero coefficient - so `has_coeffs` has to
    /// agree with the *emitted events*, not with `end_of_block_index` (see
    /// `events_from_quantized`'s doc comment on why those can diverge under
    /// trellis). This checks that agreement directly off the events trellis
    /// itself produced, across the same sweep as the cost-optimality test.
    #[test]
    fn trellis_has_coeffs_matches_emitted_events() {
        let mut rng = rand::thread_rng();
        for plane in [Plane::YCoeff1, Plane::Y2, Plane::Chroma, Plane::YCoeff0] {
            for _ in 0..2000 {
                let mut natural = [0i32; 16];
                for v in natural.iter_mut() {
                    *v = rng.gen_range(-600..=600);
                }
                let dc_quant = rng.gen_range(4..=157);
                let ac_quant = rng.gen_range(4..=284);
                let initial_context = rng.gen_range(0..=2usize);
                let lambda = 10f64.powf(rng.gen_range(-3.0..4.0));

                let (_, events, has_coeffs) = trellis_quantize_block(
                    &natural,
                    plane,
                    initial_context,
                    dc_quant,
                    ac_quant,
                    &COEFF_PROBS[plane as usize],
                    lambda,
                );

                let any_nonzero_event = events
                    .iter()
                    .any(|e| e.token != DCT_EOB && e.quantized_value != 0);
                assert_eq!(has_coeffs, any_nonzero_event);
            }
        }
    }
}
