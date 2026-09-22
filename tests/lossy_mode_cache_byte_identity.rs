//! Byte-identity regression test for image-resizer#136 (caching
//! `choose_macroblock_info`'s per-macroblock RD mode decision instead of
//! recomputing it on every pass of `encode_image`).
//!
//! # image-resizer#151 changed what this test can promise
//!
//! image-resizer#151 fixed the B_PRED entropy context (`top_b_pred`/
//! `left_b_pred`) being frozen during `count_skipped_macroblocks` /
//! `collect_token_counts`, and then extended `mb_info_cache` to also cover
//! `encode_image`'s real pass. Mode decisions and coefficients are
//! unaffected by that fix (see `mb_info_cache`'s doc comment on
//! `Vp8Encoder` and `tests/lossy_bpred_context_pixel_identity.rs`, which is
//! the dedicated regression test for that property) - but the frame
//! header's `prob_skip_false` and coefficient probabilities now describe
//! the encode that actually happens instead of one decided with a stale
//! B_PRED context, so the *entropy coding* of those same modes/coefficients
//! legitimately changed. That means the exact byte-for-byte `EXPECTED`
//! table below, captured before #151, no longer matches - by design, not by
//! regression.
//!
//! `byte_identity_matches_pre_cache_baseline` below has been narrowed
//! accordingly: it still compares against the same `EXPECTED` byte
//! reference, but only far enough to keep this file's original guarantee
//! for image-resizer#136 - that caching the RD mode decision across passes
//! never changes what gets *decoded* - by decoding the current output and
//! comparing pixels instead of comparing the container bytes wholesale.
//!
//! # Where the expected `(len, hash)` values came from
//!
//! They are the literal output of this same test file's `print_reference_values`
//! (see below), run against commit `7feed60` (`feat: adaptive quantisation (VP8
//! segmentation) in the WebP encoder`) - the commit immediately before the
//! mode-decision-caching change - with zero other changes in the tree. That is:
//!
//! 1. `git checkout 7feed60`
//! 2. this file was added, with `EXPECTED` left empty
//! 3. `cargo test --test lossy_mode_cache_byte_identity print_reference_values
//!    -- --ignored --nocapture` was run and its stdout (one `(name, len, hash)`
//!    triple per fixture) was pasted into `EXPECTED` below
//! 4. `byte_identity_matches_pre_cache_baseline` was confirmed to pass against
//!    that same `7feed60` tree (i.e. the values really do reproduce what the
//!    pre-caching code emits)
//! 5. the mode-decision cache was then implemented, and this test re-run to
//!    confirm the hashes are unchanged.
//!
//! `EXPECTED` is kept as-is (rather than re-captured post-#151) precisely
//! because it is no longer compared byte-for-byte - see above - only decoded
//! and compared at the pixel level, which #151 does not change.
//!
//! The hash is a 64-bit FNV-1a over the raw encoded WebP container bytes
//! (RIFF header included), deliberately not a cryptographic hash and not from
//! an external crate: this is a same-process regression check against a
//! value captured earlier in this same effort, not a defense against a
//! motivated adversary, so collision resistance beyond "won't happen by
//! accident for six short byte strings" buys nothing. `len` is kept
//! alongside the hash so a truncated-output bug can't hide behind a
//! coincidental hash match.

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

/// Small deterministic PRNG (xorshift32) so the "noisy" fixture is
/// reproducible without depending on the exact algorithm/version of the
/// `rand` crate (which is free to change between releases even with a fixed
/// seed) - this test's whole point is byte-for-byte reproducibility across
/// two points in time.
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

fn bytes_per_pixel(color: ColorType) -> u32 {
    match color {
        ColorType::L8 => 1,
        ColorType::La8 => 2,
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
    }
}

/// Solid-color image: every macroblock's source variance is exactly zero, so
/// this exercises `classify_segments`'s flattest quartile and the DC/skip
/// paths hardest.
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

/// Uniform pseudo-random noise: near-maximum source variance everywhere, the
/// opposite end of `classify_segments`'s activity range from `flat_fixture`,
/// and the case most likely to make B_PRED's per-submode RD search actually
/// matter (see this test module's doc comment on why that search is exactly
/// what's sensitive to the mode-decision cache change).
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

/// Mixed smooth-gradient / checkerboard image, same shape as the existing
/// `roundtrip_libwebp_lossy_segmented` unit test in `src/encoder.rs`: half
/// the frame is low-activity (drives segmentation's flattest quartile) and
/// half is high-frequency (drives the busiest quartile and is where B_PRED
/// tends to win RD mode decision), so a single fixture exercises both ends
/// of `classify_segments` plus real B_PRED submode variety.
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

/// RGBA fixture with a smooth alpha ramp (opaque on the left, transparent on
/// the right) over a noisy RGB base - exercises the lossy-with-alpha path
/// (`WebPEncoder::encode`'s `lossy_with_alpha`/`encode_alpha_lossless`
/// branch, which is a separate lossless code path this change does not
/// touch, but the whole point of hashing the full container is to catch any
/// unintended interaction).
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
        // Non-multiple-of-16, noisy.
        noisy_fixture("noisy_nonmultiple16", 131, 97, 60),
        // Non-multiple-of-16 (height), mixed low/high activity - triggers
        // segmentation and real B_PRED submode variety.
        gradient_checkerboard_fixture("gradient_checkerboard_nonmultiple16", 200, 150, 75),
        // Non-multiple-of-16, alpha channel.
        alpha_fixture("alpha_rgba_nonmultiple16", 90, 70, 50),
        // Smaller than one macroblock on both axes, non-multiple-of-16 -
        // stresses the right/bottom edge padding path.
        noisy_fixture("tiny_subblock_nonmultiple16", 10, 9, 90),
    ]
}

fn encode(fixture: &Fixture) -> Vec<u8> {
    assert_eq!(
        fixture.pixels.len() as u32,
        fixture.width * fixture.height * bytes_per_pixel(fixture.color),
        "fixture {} has mismatched pixel buffer length",
        fixture.name
    );

    let mut output = Vec::new();
    let mut encoder = WebPEncoder::new(&mut output);
    // `EncoderParams` is `#[non_exhaustive]`, so outside its own crate (as
    // this integration test is) it can't be built with struct-literal
    // syntax even via `..Default::default()` - go through `default()` and
    // mutate the (still `pub`) fields instead.
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
/// bytes per pixel depending on `has_alpha`) - what
/// `byte_identity_matches_pre_cache_baseline` below now compares, since
/// image-resizer#151 (see this file's top-level doc comment).
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

/// Not run by default (`cargo test` skips `#[ignore]`d tests). Run explicitly
/// with `cargo test --test lossy_mode_cache_byte_identity print_reference_values
/// -- --ignored --nocapture` to (re-)generate the `EXPECTED` table below -
/// see this file's top-level doc comment for how the current table was
/// produced, against `7feed60`. Still prints the raw-container hash/len,
/// not the pixel-level values `EXPECTED_PIXELS` holds - `EXPECTED` is no
/// longer asserted against directly (see the top-level doc comment) but is
/// kept for provenance/documentation.
#[test]
#[ignore]
fn print_reference_values() {
    for fixture in fixtures() {
        let bytes = encode(&fixture);
        println!(
            "(\"{}\", {}, 0x{:016x}),",
            fixture.name,
            bytes.len(),
            fnv1a_64(&bytes)
        );
    }
}

/// `(fixture name, encoded byte length, FNV-1a 64 hash of the encoded WebP
/// container)`, captured against `7feed60` - see this file's top-level doc
/// comment. Historical/documentation only as of image-resizer#151: no
/// longer compared byte-for-byte (see `EXPECTED_PIXELS` below for what is).
#[allow(dead_code)]
const EXPECTED: &[(&str, usize, u64)] = &[
    ("flat_multiple16", 82, 0xd8e81983fa17e6ce),
    ("flat_nonmultiple16", 148, 0x3a7834fff1b96815),
    ("noisy_nonmultiple16", 5410, 0xf5aa78ec203fdca9),
    (
        "gradient_checkerboard_nonmultiple16",
        1638,
        0x1b177c71b8c8d5df,
    ),
    ("alpha_rgba_nonmultiple16", 2360, 0xbee78ad5c931159f),
    ("tiny_subblock_nonmultiple16", 252, 0x924c6d1f1f20508b),
];

/// `(fixture name, decoded pixel buffer byte length, FNV-1a 64 hash of the
/// decoded pixel buffer)`. These fixtures are byte-for-byte identical
/// (name, dimensions, color type, quality, pixel generator) to
/// `tests/lossy_bpred_context_pixel_identity.rs`'s, and `EXPECTED` above
/// already proved `7feed60`'s and (pre-#151) `da8181d`'s encoded bytes were
/// identical for them - so the decoded-pixel reference values that file
/// captured against `da8181d` apply here unchanged; see its top-level doc
/// comment for exactly how they were captured.
const EXPECTED_PIXELS: &[(&str, usize, u64)] = &[
    ("flat_multiple16", 9216, 0xafc3bbf9d4331725),
    ("flat_nonmultiple16", 6201, 0x0eb251ad576ac798),
    ("noisy_nonmultiple16", 38121, 0x59f6fd8c6d7a6752),
    (
        "gradient_checkerboard_nonmultiple16",
        90000,
        0xf2fe7570cbb253bf,
    ),
    ("alpha_rgba_nonmultiple16", 25200, 0x6e41fde0cbf41f00),
    ("tiny_subblock_nonmultiple16", 270, 0x01b6e0963fad0adb),
];

/// The correctness bar for image-resizer#136: caching `choose_macroblock_info`'s
/// mode decision (instead of recomputing it on every one of `encode_image`'s
/// three passes) must not change what the encoder produces. Through
/// image-resizer#150 that was checked byte-for-byte against `EXPECTED`; as
/// of image-resizer#151 the encoded bytes legitimately differ (see this
/// file's top-level doc comment), so this now decodes both the current
/// output and checks it against `EXPECTED_PIXELS` - the pixel-level part of
/// the original guarantee, which #151 preserves.
#[test]
fn byte_identity_matches_pre_cache_baseline() {
    let fixtures = fixtures();
    assert_eq!(
        fixtures.len(),
        EXPECTED_PIXELS.len(),
        "fixture list changed size"
    );

    for (fixture, &(expected_name, expected_decoded_len, expected_pixel_hash)) in
        fixtures.iter().zip(EXPECTED_PIXELS.iter())
    {
        assert_eq!(fixture.name, expected_name, "fixture order changed");

        let bytes = encode(fixture);
        let decoded = decode(&bytes);
        let actual_pixel_hash = fnv1a_64(&decoded);

        assert_eq!(
            (decoded.len(), actual_pixel_hash),
            (expected_decoded_len, expected_pixel_hash),
            "fixture '{}' decoded to different pixels than the pre-cache baseline \
             (decoded len {} vs expected {}, pixel hash {:016x} vs expected {:016x})",
            fixture.name,
            decoded.len(),
            expected_decoded_len,
            actual_pixel_hash,
            expected_pixel_hash,
        );
    }
}
