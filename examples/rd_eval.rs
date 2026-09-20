//! Rate-distortion evaluation for the lossy VP8 encoder, against libwebp.
//!
//! Comparing encoders at the same nominal `quality` number is meaningless -
//! two encoders' "q75" are not the same quality, and can differ enough to
//! reverse the sign of a comparison. So this measures the only thing that
//! carries meaning: **how many bytes each encoder needs to reach the same
//! perceptual quality**, via a bisection on DSSIM.
//!
//! Usage:
//!   cargo run --release --example rd_eval -- <image.png>...
//!
//! Reports, per image and per DSSIM target, the size ratio ours/libwebp.
//! A ratio of 1.0 means parity; 2.0 means we spend twice the bytes for the
//! same quality.
use std::io::Cursor;

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

fn encode_libwebp(rgb: &[u8], w: u32, h: u32, q: u8) -> Vec<u8> {
    webp::Encoder::from_rgb(rgb, w, h).encode(f32::from(q)).to_vec()
}

/// Decode with libwebp for BOTH encoders, so the comparison measures the
/// bitstreams rather than two different decoders' rounding.
fn decode(bytes: &[u8]) -> (Vec<u8>, u32, u32) {
    let img = webp::Decoder::new(bytes).decode().expect("decode").to_image().to_rgb8();
    (img.as_raw().clone(), img.width(), img.height())
}

fn dssim(a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    let d = dssim_core::Dssim::new();
    let mk = |p: &[u8]| {
        let px: Vec<rgb::RGB8> =
            p.chunks_exact(3).map(|c| rgb::RGB8::new(c[0], c[1], c[2])).collect();
        d.create_image_rgb(&px, w, h).unwrap()
    };
    let (v, _) = d.compare(&mk(a), mk(b));
    v.into()
}

/// Lowest quality whose decoded DSSIM is <= `target`, with its byte size.
/// Returns None when even q=100 cannot reach the target.
fn search(target: f64, orig: &[u8], w: u32, h: u32, ours: bool) -> Option<(u8, usize)> {
    let (mut lo, mut hi) = (1u8, 100u8);
    let mut best = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let bytes = if ours { encode_ours(orig, w, h, mid) } else { encode_libwebp(orig, w, h, mid) };
        let (dec, dw, dh) = decode(&bytes);
        let d = dssim(orig, &dec, dw as usize, dh as usize);
        if d <= target {
            best = Some((mid, bytes.len()));
            if mid == 1 { break }
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    best
}

fn main() {
    let files: Vec<String> = std::env::args().skip(1).collect();
    assert!(!files.is_empty(), "usage: rd_eval <image.png>...");
    const TARGETS: [f64; 3] = [0.0150, 0.0080, 0.0035];

    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); TARGETS.len()];
    println!("{:<16} {:>9} {:>12} {:>12} {:>8}", "image", "DSSIM<=", "ours", "libwebp", "ratio");
    for f in &files {
        let img = image::open(f).expect("open").to_rgb8();
        let (w, h) = (img.width(), img.height());
        let orig = img.as_raw();
        let name = f.rsplit('/').next().unwrap();
        for (i, t) in TARGETS.iter().enumerate() {
            match (search(*t, orig, w, h, true), search(*t, orig, w, h, false)) {
                (Some((qo, so)), Some((ql, sl))) => {
                    let r = so as f64 / sl as f64;
                    ratios[i].push(r);
                    println!("{name:<16} {t:>9.4} q{qo:<3}{so:>8} q{ql:<3}{sl:>8} {r:>7.2}x");
                }
                _ => println!("{name:<16} {t:>9.4} {:>12} {:>12} {:>8}", "unreachable", "-", "-"),
            }
        }
    }

    println!("\n=== median size ratio (ours / libwebp), lower is better ===");
    for (i, t) in TARGETS.iter().enumerate() {
        let v = &mut ratios[i];
        if v.is_empty() { continue }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  DSSIM <= {:.4}   n={:<3} median {:.3}x", t, v.len(), v[v.len() / 2]);
    }
}
