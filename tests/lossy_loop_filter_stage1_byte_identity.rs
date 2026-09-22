//! Byte-identity regression test for image-resizer#137 stage 1 (plumbing
//! full Y/U/V reconstruction planes into `Vp8Encoder`, and sourcing
//! intra-prediction borders from them instead of the incremental
//! `top_border_*`/`left_border_*` caches those planes replace).
//!
//! # The property this checks
//!
//! Stage 1 is a pure refactor of the *plumbing* - full reconstruction
//! planes, borders sourced from them - and is only guaranteed
//! byte-identical to `817951e` while `frame.filter_level` is `0` (no
//! filtering applied). Stage 2 (image-resizer#137's second half) derives a
//! generally-nonzero `frame.filter_level` from the quantiser
//! (`derive_filter_level`) and actually applies it, so this test can no
//! longer encode at an arbitrary quality and expect byte-identical output -
//! that is exactly the property `tests/lossy_loop_filter_drift.rs` covers
//! instead, where decoded pixels are *expected* to change.
//!
//! What this test keeps covering, permanently, is the plumbing in
//! isolation: every fixture below uses `lossy_quality: 100`, which makes
//! `setup_encoding` derive a base quantiser index (`quant_index`) of
//! exactly `0` - and `derive_filter_level(0)` is `0` regardless of
//! `FILTER_LEVEL_DIVISOR`, since `0` divided by anything is `0`. So these
//! fixtures exercise the reconstruction-plane plumbing (borders, partial
//! edge macroblocks, segmentation, B_PRED) with the loop filter
//! structurally inert, the same guarantee stage 1 originally shipped with -
//! without pinning this test to `derive_filter_level`'s specific curve.
//!
//! Any difference here means the plumbing itself is wrong (e.g. a border
//! sourced from the wrong plane offset), independent of anything loop-filter
//! related.
//!
//! # Where the expected `(len, hash)` values came from
//!
//! They are the literal output of this same test file's
//! `print_reference_values` (see below), run against commit `817951e`
//! (`perf(webp): bump the vaam-image-webp fork to 817951e`) with zero other
//! changes in the tree, using a temporary `git worktree` checked out at that
//! commit (this crate's `main` had already moved on by the time these
//! fixtures were switched to `lossy_quality: 100`):
//!
//! 1. `git worktree add <tmp-dir> 817951e`
//! 2. this file (with today's `lossy_quality: 100` fixtures and `EXPECTED`
//!    left empty) was copied into `<tmp-dir>/tests/`
//! 3. `cargo test --test lossy_loop_filter_stage1_byte_identity \
//!    print_reference_values -- --ignored --nocapture` was run there and
//!    its stdout (one `(name, len, hash)` triple per fixture) was pasted
//!    into `EXPECTED` below
//! 4. the temporary worktree was removed (`git worktree remove <tmp-dir>`)
//! 5. back on the real tree, with stage 1's plumbing already implemented
//!    (`frame.filter_level` still hardcoded to `0` at that point) and later
//!    stage 2's `derive_filter_level` wired in, this test was confirmed to
//!    still pass - proving the quality-100 fixtures really do stay
//!    filter-inert across both stages.
//!
//! The hash is a 64-bit FNV-1a over the raw encoded WebP container bytes
//! (RIFF header included) - not a cryptographic hash, see the equivalent
//! comment in `tests/lossy_mode_cache_byte_identity.rs` for why that's fine
//! here (same-process regression check, not an adversarial setting).

use image_webp::{ColorType, EncoderParams, WebPEncoder};

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
/// type in `lossy_mode_cache_byte_identity.rs`.
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

/// Solid-color image: every macroblock's source variance is exactly zero,
/// so every macroblock's borders come from the flattest, most
/// default-value-heavy prediction path.
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

/// Uniform pseudo-random noise: near-maximum source variance everywhere,
/// which drives real B_PRED submode variety and therefore exercises the
/// new plane-sourced borders hardest (every sub-block boundary within a
/// macroblock reads a just-written border).
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

/// Mixed smooth-gradient / checkerboard image, same shape as the fixture in
/// `lossy_mode_cache_byte_identity.rs`: exercises both ends of
/// `classify_segments` plus real B_PRED submode variety in one fixture.
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

/// RGBA fixture with a smooth alpha ramp over a noisy RGB base.
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

// `lossy_quality: 100` on every fixture below is load-bearing - see this
// file's top-level doc comment for why that, not the geometry, is what
// keeps the loop filter structurally inert here.
fn fixtures() -> Vec<Fixture> {
    vec![
        // Multiple-of-16 dimensions, flat.
        flat_fixture("flat_multiple16", 64, 48, 100),
        // Non-multiple-of-16 dimensions (both axes), flat - exercises the
        // partial edge macroblocks the reconstruction planes must size for.
        flat_fixture("flat_nonmultiple16", 53, 39, 100),
        // Non-multiple-of-16, noisy.
        noisy_fixture("noisy_nonmultiple16", 131, 97, 100),
        // Non-multiple-of-16 (height), mixed low/high activity.
        gradient_checkerboard_fixture("gradient_checkerboard_nonmultiple16", 200, 150, 100),
        // Non-multiple-of-16, alpha channel.
        alpha_fixture("alpha_rgba_nonmultiple16", 90, 70, 100),
        // Smaller than one macroblock on both axes, non-multiple-of-16 -
        // stresses the right/bottom edge padding path hardest: this whole
        // image is a single partial macroblock.
        noisy_fixture("tiny_subblock_nonmultiple16", 10, 9, 100),
        // A single macroblock, exactly 16x16 - the degenerate case with no
        // neighbours on any side.
        noisy_fixture("single_macroblock", 16, 16, 100),
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

/// Not run by default. Run explicitly, against `817951e`, to (re-)generate
/// `EXPECTED` - see this file's top-level doc comment for the exact command
/// and provenance.
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
/// container)`, captured against `817951e` - see this file's top-level doc
/// comment.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("flat_multiple16", 94, 0x1c1c35ecf2d911c1),
    ("flat_nonmultiple16", 244, 0xf022ad9e29f0c4fa),
    ("noisy_nonmultiple16", 14732, 0x3ffda3d4cb7c5f6a),
    (
        "gradient_checkerboard_nonmultiple16",
        3820,
        0x572d9d26db1bad26,
    ),
    ("alpha_rgba_nonmultiple16", 8312, 0x692a27c59f79ca59),
    ("tiny_subblock_nonmultiple16", 316, 0x788cca14137c99e3),
    ("single_macroblock", 600, 0xb8ae48f6213c6d13),
];

/// The correctness bar for image-resizer#137 stage 1: plumbing full Y/U/V
/// reconstruction planes into the encoder, and sourcing intra-prediction
/// borders from them instead of the old incremental caches, must not change
/// a single byte of the encoded output while `frame.filter_level` is still
/// `0` - see this file's top-level doc comment.
#[test]
fn stage1_plumbing_is_byte_identical_to_817951e() {
    let fixtures = fixtures();
    assert_eq!(fixtures.len(), EXPECTED.len(), "fixture list changed size");

    for (fixture, &(expected_name, expected_len, expected_hash)) in
        fixtures.iter().zip(EXPECTED.iter())
    {
        assert_eq!(fixture.name, expected_name, "fixture order changed");

        let bytes = encode(fixture);
        let actual_hash = fnv1a_64(&bytes);

        assert_eq!(
            (bytes.len(), actual_hash),
            (expected_len, expected_hash),
            "fixture '{}': stage-1 plumbing changed the encoded bytes (len {} vs expected \
             {}, hash {:016x} vs expected {:016x}) - with filter_level still 0 this must be \
             byte-for-byte identical to 817951e",
            fixture.name,
            bytes.len(),
            expected_len,
            actual_hash,
            expected_hash,
        );
    }
}
