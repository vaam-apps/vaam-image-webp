# vaam-image-webp

[![crates.io](https://img.shields.io/crates/v/vaam-image-webp.svg)](https://crates.io/crates/vaam-image-webp)
[![Documentation](https://docs.rs/vaam-image-webp/badge.svg)](https://docs.rs/vaam-image-webp)
[![Build Status](https://github.com/vaam-apps/vaam-image-webp/workflows/Rust%20CI/badge.svg)](https://github.com/vaam-apps/vaam-image-webp/actions)

`vaam-image-webp` is a pure-Rust WebP codec. As far as we know, it is the only
published crate that encodes **lossy** (VP8) WebP without linking `libwebp` or
any other C/C++ code — every other pure-Rust option we're aware of,
including upstream `image-webp` 0.2.4, only encodes losslessly, and lossless
WebP runs roughly **5-12x larger** than lossy on photographic content.

Use it when you need WebP output and cannot or would rather not link C:
sandboxed environments, cross-compilation targets where `libwebp` is a
headache, supply-chain policies that forbid C dependencies, or WASM.

## The trade-off

This is not a faster or smaller alternative to libwebp — it's what you reach
for when linking libwebp isn't an option. Measured on the 24-image Kodak
suite, lossy output from this crate compared against libwebp:

* **File size:** 1.17-1.21x libwebp's size at matched DSSIM (i.e. our files
  are 17-21% larger for the same perceptual quality).
* **Encode speed:** ~2.7x libwebp's encode time (i.e. we are slower, not
  faster).
* **Decoder conformance:** 72/72 bit-exact against libwebp on the same suite.

You're trading bytes and CPU for not linking C. If neither of those trades
work for your use case, use `libwebp` (or the `webp` crate, which wraps it)
instead.

## Current status

* **Decoder:** supports all WebP format features — lossless, lossy, alpha
  channel, and animation, both "simple" and "extended" formats — and exposes
  methods to extract ICC, EXIF, and XMP chunks. Decoding speed is generally
  70-100% of libwebp's.

* **Encoder:** supports both lossless and lossy (VP8) encoding. Lossless
  encoding is unchanged from upstream: fast, simple, and often smaller than
  PNG even against the slowest PNG encoders, though not as tight as libwebp.
  Lossy encoding is this crate's addition over upstream — see the trade-off
  numbers above for how it compares.

## Usage

```rust
use image_webp::{WebPEncoder, WebPDecoder};
```

Note the crate is `vaam-image-webp` but the library is still named
`image_webp` (see [Relationship to upstream](#relationship-to-upstream)), so
`Cargo.toml` and source imports look like:

```toml
[dependencies]
vaam-image-webp = "0.1"
```

See [docs.rs](https://docs.rs/vaam-image-webp) for the full API.

## Relationship to upstream

This crate is a fork of [`image-rs/image-webp`](https://github.com/image-rs/image-webp),
maintained by [vaam-apps](https://github.com/vaam-apps). It exists because
upstream's lossy VP8 encoder has not shipped in a crates.io release: `main`
has one, `0.2.4` doesn't. Rather than depend on an upstream git SHA
indefinitely, we publish it as its own crate with its own version line,
starting at `0.1.0`.

What we do:

* Track upstream's decoder and lossless encoder as closely as practical.
* Add and maintain the lossy (VP8) encoder — adaptive quantisation, RD-aware
  mode decisions, and the arithmetic coder that goes with them.
* Upstream bug fixes we find (decoder or shared code) go back to upstream via
  PR, not just fixed here in isolation.

The Rust lib target is still named `image_webp`, matching upstream, so this
crate is a drop-in replacement — swapping the dependency line is enough, no
import changes needed.

Everything not called out above is upstream's work, licensed under upstream's
terms (see below). Thanks to the `image-rs` maintainers and contributors for
the decoder, the lossless encoder, and the format work this crate builds on.

## Unsafe code

Both this crate and all of its dependencies currently contain no unsafe code.

NOTE: This isn't a guarantee that unsafe code will never be added. It may prove
necessary in the future to improve performance, but we will always strive to
minimize the use of unsafe code and ensure that it is well-tested and
documented.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option, matching upstream
`image-rs/image-webp`'s licensing.
