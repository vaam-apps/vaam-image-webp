# Release Notes

## vaam-image-webp

This crate is a fork of upstream `image-webp`; see [README.md](README.md#relationship-to-upstream)
for what it adds and how it relates to upstream. Its own version line starts
below. The `Version 0.2.x` and earlier entries further down are upstream
`image-webp`'s changelog, kept as-is for history and unaffected by anything
in this section.

### Version 0.1.0

First release of `vaam-image-webp` as its own crate on crates.io. Forked from
upstream `image-webp` (last synced against its `0.2.4` release plus
unreleased `main`).

Changes over upstream:
 - Added a lossy (VP8) encoder: pure Rust, no C/C++ dependencies. Upstream's
   `0.2.4` only encodes losslessly.
 - Adaptive quantisation (VP8 segmentation) and RD-aware dry-run mode
   decisions in the lossy encoder.
 - Measured against libwebp on the 24-image Kodak suite: 1.17-1.21x file
   size at matched DSSIM, ~2.7x encode time, and the decoder is 72/72
   bit-exact.

# Upstream Release Notes (image-webp)

### Version 0.2.4

Changes:
 - Changed default upscaling to bilinear interpolation to match libwebp (#147)

Bug fixes:
 - Fixed all remaining divergences against libwebp in loop filtering (#148, #149)

Optimizations:
 - Optimized predictors in lossless_transform (#152)
 - Improved performance of horizontal loop filtering (#151, #156)


### Version 0.2.3

Changes:
 - Do not reject images with ICC profile bit set but missing ICCP chunk (#143)

Bug Fixes:
 - Fixed a bug that caused the last chroma macroblock in the image to be sometimes decoded incorrectly (#144)

### Version 0.2.2

Changes:
 - Do not apply background color to animated images by default to better match libwebp behavior (#135)

Bug Fixes:
 - Fixed a bug in the loop filter causing subtly but noticeably incorrect decoding of some lossy images (#140)

Optimizations:
 - Remove bounds checks from color transform hot loop (#133)
 - Optimize resolving indexed images into RGB colors (#132, #134)

### Version 0.2.1

Changes:
 - Increased the required Rust compiler version to v1.80

Optimizations:
 - Removed bounds checks from hot loops in `read_coefficients()` (#121)
 - Faster YUV -> RGBA conversion for a 7% speedup on lossy RGBA images (#122)
 - Faster alpha blending for up to 20% speedup on animated images (#123)
 - Much faster arithmetic decoding for up to 30% speedup on lossy images (#124)
 - Avoid unnecessarily cloning image data for a 4% speedup (#126)

### Version 0.2.0

Breaking Changes:
- `WebPDecoder` now requires the passed reader implement `BufRead`.

Changes:
- Add `EncoderParams` to make predictor transform optional.

Bug Fixes:
- Several bug fixes in animation compositing.
- Fix indexing for filling image regions with tivial huffman codes.
- Properly update the color cache when trivial huffman codes are used.

Optimizations:
- Substantially faster decoding of lossless images, by switching to a
  table-based Huffman decoder and a variety of smaller optimizations.

### Version 0.1.3

Changes:
- Accept files with out-of-order "unknown" chunks.
- Switched to `quick-error` crate for faster compliation.

Bug Fixes:
- Fixed decoding of animations with ALPH chunks.
- Fixed encoding bug for extended WebP files.
- Resolved compliation problem with fuzz targets.

Optimizations:
- Faster YUV to RGB conversion.
- Improved `BitReader` logic.
- In-place decoding of lossless RGBA images.

### Version 0.1.2

- Export `decoder::LoopCount`.
- Fix decode bug in `read_quantization_indices` that caused some lossy images to
  appear washed out.
- Switch to `byteorder-lite` crate for byte order conversions.

### Version 0.1.1

- Fix RIFF size calculation in encoder.

### Version 0.1.0

- Initial release
