# fast-thumbhash

A **12x faster** drop-in replacement for the [`thumbhash`](https://crates.io/crates/thumbhash) crate — with built-in base91 encoding that beats BlurHash on every axis.

[ThumbHash](https://evanw.github.io/thumbhash/) is a compact image placeholder algorithm — similar to BlurHash but with better detail, color accuracy, and alpha support. This crate implements the same algorithm with aggressive low-level optimizations, producing perceptually identical results at a fraction of the cost.

## Performance

Benchmarked on Apple M-series, 100x75 RGBA input, 10,000 iterations:

| Operation | `thumbhash` | `fast-thumbhash` | Speedup |
|-----------|-------------|-------------------|---------|
| Encode    | 172.7 µs    | 14.4 µs           | **12.0x** |
| Decode    | 18.3 µs     | 1.5 µs            | **12.4x** |

## vs BlurHash

`fast-thumbhash` with base91 encoding produces **smaller strings** than BlurHash's base83 while carrying strictly more information:

| | BlurHash (base83) | fast-thumbhash (base91) |
|---|---|---|
| **Encode speed** | ~180 µs | **14 µs** (12x faster) |
| **Typical string size** | 28 chars | **26 chars** |
| **Alpha channel** | No | Yes |
| **Aspect ratio** | No | Embedded |
| **Detail** | 4x3 RGB grid | Full DCT L/P/Q/A |

More data in fewer characters — faster to generate, smaller on the wire, better previews.

## Base91 encoding

The crate includes a built-in base91 encoder/decoder — the most efficient text encoding possible for JSON-safe ASCII strings (only ~23% overhead vs base64's 33%).

The base91 alphabet excludes `"`, `\`, and `-`, so output is safe for JSON strings without escaping.

```rust
use fast_thumbhash::{base91_encode, base91_decode};

let encoded = base91_encode(&bytes);       // compact ASCII string
let decoded = base91_decode(&encoded)?;    // back to bytes

// Or use the convenience functions:
use fast_thumbhash::{rgba_to_thumb_hash_b91, thumb_hash_from_b91};

let hash_str = rgba_to_thumb_hash_b91(w, h, &rgba);    // RGBA → base91 string
let (w, h, pixels) = thumb_hash_from_b91(&hash_str)?;  // base91 string → RGBA
```

## How it's faster

**Encoder:**
- **Separable 2D DCT** — splits the O(W\*H\*N) transform into two 1D passes, cutting multiply-adds by ~3x
- **Chebyshev cosine recurrence** — computes cosine tables with 2 `cos()` calls per frequency instead of one per pixel
- **Stack-allocated buffers** — cos tables and partial sums live on the stack, zero heap allocation in the hot path
- **Integer averaging** — computes the average color in integer space, avoiding N float divisions
- **Opaque fast path** — skips alpha compositing entirely for fully-opaque images (most photos)
- **Branchless nibble packing** — collects AC coefficients then packs pairs without per-nibble branching

**Decoder:**
- **Separable 2D IDCT** — precomputes x-contributions per frequency band, then accumulates rows via SAXPY (auto-vectorizes to NEON/SSE)
- **Stack-allocated everything** — cos tables, AC buffers, and scratch rows all on the stack
- **Direct nibble indexing** — reads AC data by index instead of through `std::io::Read`

Both paths use `unsafe get_unchecked` to eliminate bounds checks in inner loops.

## Compatibility

`fast-thumbhash` is a **perceptually identical** replacement — not bit-identical. The separable DCT and Chebyshev recurrence change the floating-point evaluation order, producing sub-nibble rounding differences in some AC coefficients.

In practice, across a library of real-world images:
- **PSNR > 70 dB** between decoded previews (60 dB is already indistinguishable to human eyes)
- Maximum pixel delta of 2-3/255 in affected channels
- Headers (dimensions, average color, structure) are always identical

The raw byte API is the same as `thumbhash` — swap the crate name and everything works.

## Usage

```toml
[dependencies]
fast-thumbhash = "0.2"
```

```rust
use fast_thumbhash::{rgba_to_thumb_hash, thumb_hash_to_rgba};

// Encode: RGBA pixels → compact hash bytes
let hash = rgba_to_thumb_hash(width, height, &rgba_pixels);

// Decode: hash bytes → RGBA preview image
let (w, h, pixels) = thumb_hash_to_rgba(&hash).unwrap();

// Base91 string encoding (recommended for storage/transport)
use fast_thumbhash::{base91_encode, base91_decode};
let encoded = base91_encode(&hash);  // e.g. "}U#WoBrZy#_/qQ8PC,N]q7m}6X"
let bytes = base91_decode(&encoded).unwrap();

// Utilities
let (r, g, b, a) = fast_thumbhash::thumb_hash_to_average_rgba(&hash).unwrap();
let aspect = fast_thumbhash::thumb_hash_to_approximate_aspect_ratio(&hash).unwrap();
```

## License

MIT
