//! Fixed-quality evaluation for the lossy VP8 encoder: bytes and SSIMULACRA2
//! at q40/q55/q70/q85, against the encoder's own libwebp-decoded output.
//!
//! `rd_eval` (DSSIM-bisected, against libwebp) answers "how many bytes do we
//! need for the same quality libwebp reaches" - useful, but adaptive
//! quantisation is a perceptual optimisation that redistributes error rather
//! than removing it, so it can look flat or even negative on a structural
//! metric (DSSIM, SSE) while being a real perceptual win. SSIMULACRA2 is
//! trained to track human perception more closely than either, so this
//! harness holds `lossy_quality` fixed and reports both bytes and
//! SSIMULACRA2 - the pair that actually decides whether a perceptual change
//! helped.
//!
//! Usage:
//!   cargo run --release --example ssimu2_eval -- <image.png>...
use std::io::Cursor;

use ssimulacra2::{compute_frame_ssimulacra2, ColorPrimaries, Rgb, TransferCharacteristic};

const QUALITIES: [u8; 4] = [40, 55, 70, 85];

fn encode_ours(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    let mut out = Vec::new();
    let mut e = image_webp::WebPEncoder::new(Cursor::new(&mut out));
    let mut p = image_webp::EncoderParams::default();
    p.use_lossy = true;
    p.lossy_quality = q;
    e.set_params(p);
    e.encode(rgb, w, h, image_webp::ColorType::Rgb8).unwrap();
    out
}

/// Decode with libwebp - the correctness gate for this crate's bitstreams
/// (see `rd_eval`) and, incidentally, what most real-world consumers of a
/// `.webp` file will use.
fn decode(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = webp::Decoder::new(bytes)
        .decode()
        .expect("libwebp failed to decode our bitstream")
        .to_image()
        .to_rgb8();
    (img.as_raw().clone(), img.width(), img.height())
}

fn to_ssimulacra2_rgb(px: &[u8], w: u32, h: u32) -> Rgb {
    let data: Vec<[f32; 3]> = px
        .chunks_exact(3)
        .map(|c| {
            [
                f32::from(c[0]) / 255.0,
                f32::from(c[1]) / 255.0,
                f32::from(c[2]) / 255.0,
            ]
        })
        .collect();
    Rgb::new(
        data,
        w as usize,
        h as usize,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .expect("pixel count must match width * height")
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "usage: ssimu2_eval <image.png>...");

    println!(
        "{:<12} {:>5} {:>10}  {:>8}",
        "image", "q", "bytes", "SSIMU2"
    );
    for f in &files {
        let img = image::open(f).expect("open").to_rgb8();
        let (w, h) = (img.width(), img.height());
        let orig = img.as_raw();
        let name = f.rsplit('/').next().unwrap();
        let src = to_ssimulacra2_rgb(orig, w, h);

        for &q in &QUALITIES {
            let bytes = encode_ours(orig, w, h, q);
            let (dec, dw, dh) = decode(&bytes);
            assert_eq!((dw, dh), (w, h), "decoded dimensions must match the source");
            let dist = to_ssimulacra2_rgb(&dec, dw, dh);
            let score = compute_frame_ssimulacra2(src.clone(), dist).expect("ssimulacra2 compute");
            println!("{name:<12} q{q:<4} {:>7} B  {:>8.3}", bytes.len(), score);
        }
    }
}
