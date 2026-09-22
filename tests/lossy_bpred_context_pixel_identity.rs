//! Pixel-identity / size regression test for image-resizer#151 (keeping the
//! B_PRED entropy context in sync across `encode_image`'s two dry runs, then
//! extending `mb_info_cache` to also cover its real pass).
//!
//! # The property this checks
//!
//! Before this change, `count_skipped_macroblocks` and `collect_token_counts`
//! made mode decisions with `top_b_pred`/`left_b_pred` frozen at their
//! `reset_frame_state` defaults, so the `prob_skip_false` and coefficient
//! probabilities the frame header advertises were derived from a B_PRED
//! context the real pass never actually has. The real pass's own mode
//! decisions never depended on the dry runs, so this change does not alter
//! them - only the header statistics describing how those decisions are
//! entropy-coded. That means:
//!
//! - decoded pixels must come out bit-identical to what the pre-fix encoder
//!   (`da8181d`) produced (same modes, same coefficients - only their
//!   entropy coding differs), and
//! - the encoded size should not grow, since the probabilities now describe
//!   the encode that actually happens rather than a frozen-context guess.
//!
//! # Where the `EXPECTED` values came from
//!
//! Same fixtures as `tests/lossy_mode_cache_byte_identity.rs` (see that
//! file's own doc comment for the general pattern this follows). Captured
//! against commit `da8181d` (`perf: bump the vaam-image-webp fork to
//! da8181d (image-resizer#153)`, i.e. this crate's `main` immediately before
//! this fix) via a temporary copy of this file's `print_reference_values`
//! (below) added to a `git worktree` checked out at `da8181d`, run with:
//!
//! ```text
//! cargo test --test lossy_bpred_context_pixel_identity print_reference_values \
//!     -- --ignored --nocapture
//! ```
//!
//! Its stdout - one `(name, encoded_len, decoded_len, decoded_pixel_hash)`
//! quadruple per fixture - was pasted into `EXPECTED` below verbatim, and the
//! temporary worktree/test file were discarded afterwards; nothing here runs
//! against `da8181d` at test time. `decoded_len` (the decoded pixel buffer's
//! byte length) is kept alongside the hash so a truncated- or wrong-sized-
//! decode bug can't hide behind a coincidental hash match, mirroring why
//! `lossy_mode_cache_byte_identity.rs` keeps `len` next to its hash.

use std::io::Cursor;

use image_webp::{ColorType, EncoderParams, WebPDecoder, WebPEncoder};

fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Small deterministic PRNG (xorshift32), same rationale as the equivalent
/// type in `lossy_mode_cache_byte_identity.rs`: reproducible fixtures
/// without depending on the exact behaviour of the `rand` crate.
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

struct Fixture {
    name: &'static str,
    width: u32,
    height: u32,
    color: ColorType,
    lossy_quality: u8,
    pixels: Vec<u8>,
}

/// Solid-color image: every macroblock's source variance is exactly zero -
/// exercises `classify_segments`'s flattest quartile and the DC/skip paths
/// hardest.
fn flat_fixture(name: &'static str, width: u32, height: u32, lossy_quality: u8) -> Fixture {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for chunk in pixels.chunks_exact_mut(3) {
        chunk.copy_from_slice(&[40, 90, 160]);
    }
    Fixture {
        name,
        width,
        height,
        color: ColorType::Rgb8,
        lossy_quality,
        pixels,
    }
}

/// Uniform pseudo-random noise: near-maximum source variance everywhere -
/// the case most likely to make B_PRED's per-submode RD search (and
/// therefore the entropy context this fix corrects) actually matter.
fn noisy_fixture(name: &'static str, width: u32, height: u32, lossy_quality: u8) -> Fixture {
    let mut rng = Xorshift32(0x1234_5678 ^ (width << 16) ^ height);
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for b in pixels.iter_mut() {
        *b = rng.next_u8();
    }
    Fixture {
        name,
        width,
        height,
        color: ColorType::Rgb8,
        lossy_quality,
        pixels,
    }
}

/// Mixed smooth-gradient / checkerboard image: half the frame is
/// low-activity, half is high-frequency, so a single fixture exercises both
/// ends of `classify_segments` plus real B_PRED submode variety.
fn gradient_checkerboard_fixture(
    name: &'static str,
    width: u32,
    height: u32,
    lossy_quality: u8,
) -> Fixture {
    let mut pixels = vec![0u8; (width * height * 3) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 3) as usize;
            let (r, g, b) = if x < width / 2 {
                let v = ((x * 255) / width.max(1)) as u8;
                (v, v.wrapping_add((y % 255) as u8), 128)
            } else {
                let v = if (x / 4 + y / 4) % 2 == 0 { 20 } else { 235 };
                (v, v, v)
            };
            pixels[i] = r;
            pixels[i + 1] = g;
            pixels[i + 2] = b;
        }
    }
    Fixture {
        name,
        width,
        height,
        color: ColorType::Rgb8,
        lossy_quality,
        pixels,
    }
}

/// RGBA fixture with a smooth alpha ramp over a noisy RGB base - exercises
/// the lossy-with-alpha path alongside the change under test.
fn alpha_fixture(name: &'static str, width: u32, height: u32, lossy_quality: u8) -> Fixture {
    let mut rng = Xorshift32(0x9e37_79b9 ^ (width << 16) ^ height);
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            pixels[i] = rng.next_u8();
            pixels[i + 1] = rng.next_u8();
            pixels[i + 2] = rng.next_u8();
            pixels[i + 3] = ((x * 255) / width.max(1)) as u8;
        }
    }
    Fixture {
        name,
        width,
        height,
        color: ColorType::Rgba8,
        lossy_quality,
        pixels,
    }
}

fn fixtures() -> Vec<Fixture> {
    vec![
        // Multiple-of-16 dimensions, flat.
        flat_fixture("flat_multiple16", 64, 48, 80),
        // Non-multiple-of-16 dimensions (both axes), flat.
        flat_fixture("flat_nonmultiple16", 53, 39, 40),
        // Non-multiple-of-16, noisy - B_PRED-heavy.
        noisy_fixture("noisy_nonmultiple16", 131, 97, 60),
        // Non-multiple-of-16 (height), mixed low/high activity.
        gradient_checkerboard_fixture("gradient_checkerboard_nonmultiple16", 200, 150, 75),
        // Non-multiple-of-16, alpha channel.
        alpha_fixture("alpha_rgba_nonmultiple16", 90, 70, 50),
        // Smaller than one macroblock on both axes.
        noisy_fixture("tiny_subblock_nonmultiple16", 10, 9, 90),
    ]
}

fn encode(fixture: &Fixture) -> Vec<u8> {
    let mut output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut output);
    // `EncoderParams` is `#[non_exhaustive]` - see
    // `lossy_mode_cache_byte_identity.rs` for why this goes through
    // `default()` + field mutation rather than struct-literal syntax.
    let mut params = EncoderParams::default();
    params.use_lossy = true;
    params.lossy_quality = fixture.lossy_quality;
    encoder.set_params(params);
    encoder
        .encode(
            &fixture.pixels,
            fixture.width,
            fixture.height,
            fixture.color,
        )
        .unwrap_or_else(|e| panic!("encode failed for fixture {}: {e}", fixture.name));
    output
}

/// Decodes a WebP container back to a raw RGB/RGBA pixel buffer (3 or 4
/// bytes per pixel depending on `has_alpha`).
fn decode(bytes: &[u8]) -> Vec<u8> {
    let mut decoder = WebPDecoder::new(Cursor::new(bytes)).expect("produced a valid webp");
    let (width, height) = decoder.dimensions();
    let bytes_per_pixel = if decoder.has_alpha() { 4 } else { 3 };
    let mut data = vec![0u8; width as usize * height as usize * bytes_per_pixel];
    decoder
        .read_image(&mut data)
        .expect("decode the image we just encoded");
    data
}

/// Not run by default. Run explicitly, against `da8181d`, to (re-)generate
/// `EXPECTED` - see this file's top-level doc comment for the exact command
/// and provenance.
#[test]
#[ignore]
fn print_reference_values() {
    for fixture in fixtures() {
        let encoded = encode(&fixture);
        let decoded = decode(&encoded);
        println!(
            "(\"{}\", {}, {}, 0x{:016x}),",
            fixture.name,
            encoded.len(),
            decoded.len(),
            fnv1a_64(&decoded)
        );
    }
}

/// `(fixture name, da8181d encoded byte length, da8181d decoded pixel buffer
/// byte length, FNV-1a 64 hash of da8181d's decoded pixel buffer)`, captured
/// against `da8181d` - see this file's top-level doc comment.
const EXPECTED: &[(&str, usize, usize, u64)] = &[
    ("flat_multiple16", 82, 9216, 0xafc3bbf9d4331725),
    ("flat_nonmultiple16", 148, 6201, 0x0eb251ad576ac798),
    ("noisy_nonmultiple16", 5410, 38121, 0x59f6fd8c6d7a6752),
    (
        "gradient_checkerboard_nonmultiple16",
        1638,
        90000,
        0xf2fe7570cbb253bf,
    ),
    ("alpha_rgba_nonmultiple16", 2360, 25200, 0x6e41fde0cbf41f00),
    ("tiny_subblock_nonmultiple16", 252, 270, 0x01b6e0963fad0adb),
];

/// The correctness bar for image-resizer#151: with the B_PRED entropy
/// context now correctly tracked during the dry runs (and `mb_info_cache`
/// extended to the real pass on top of that), decoded pixels were
/// bit-identical to what `da8181d` produced, and the encoded size did not
/// grow.
///
/// # image-resizer#137 superseded both halves of that guarantee
///
/// The loop filter (image-resizer#137 stage 2) applies a generally nonzero
/// `frame.filter_level` (`derive_filter_level`, keyed off the quantiser) to
/// the encoder's own reconstruction, and a real decoder applies the
/// matching filter when decoding this crate's output - both legitimately
/// change decoded pixels relative to `EXPECTED` (captured when the loop
/// filter was unconditionally disabled), and change encoded size in
/// whichever direction the filtered residual actually costs (the frame
/// header alone now spends more bits signalling a nonzero level, sharpness
/// and filter type, on top of whatever the filtered reconstruction changes
/// about the residual itself) - "must not grow" is no longer a property
/// this change preserves either. `EXPECTED`'s pixel hash and the old
/// never-grows assertion are dropped for that reason; the decoded-length
/// check remains (a truncated/malformed decode is still a real
/// regression), and `Vp8Encoder::apply_loop_filter`'s own drift test
/// (`src/lossy/encoder.rs`'s `tests` module) is what now covers "decoded
/// pixels are what they should be" for loop-filter-affected output.
#[test]
fn bpred_context_fix_keeps_pixel_buffer_size_stable() {
    let fixtures = fixtures();
    assert_eq!(fixtures.len(), EXPECTED.len(), "fixture list changed size");

    for (fixture, &(expected_name, old_len, expected_decoded_len, _expected_pixel_hash_pre_137)) in
        fixtures.iter().zip(EXPECTED.iter())
    {
        assert_eq!(fixture.name, expected_name, "fixture order changed");

        let new_bytes = encode(fixture);
        let new_decoded = decode(&new_bytes);

        assert_eq!(
            new_decoded.len(),
            expected_decoded_len,
            "fixture '{}': decoded to a different-sized pixel buffer than da8181d (decoded \
             len {} vs expected {})",
            fixture.name,
            new_decoded.len(),
            expected_decoded_len,
        );

        let old_to_new_pct = 100.0 * (new_bytes.len() as f64 - old_len as f64) / old_len as f64;
        eprintln!(
            "{}: {} -> {} bytes ({:+.2}%)",
            fixture.name,
            old_len,
            new_bytes.len(),
            old_to_new_pct,
        );
    }
}
