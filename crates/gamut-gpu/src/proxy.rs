//! The reference proxy of a source on the GPU: what the dabs of an auto
//! stroke read their reference colour from. gamut-color's `Proxy` is its
//! twin.
//!
//! The proxy is the whole source reduced by a box filter so its longer side
//! is 1024 pixels, in linear Rec.2020, kept in the working format. It is a
//! product of the source and of nothing else: it is built when the source
//! content changes, and a render of any window reads the same one, so the
//! fitted view, a zoomed window and the export agree on every reference.
//!
//! A large source is not held at its full size in the working format (24
//! megapixels would take 192 MB). It is drawn a tile at a time, at its own
//! size, by the pass every render reads it through, and `proxy.wgsl`
//! averages each tile into the proxy pixels it serves. [`tiles`] cuts the
//! proxy into blocks whose source pixels fit a tile.

use bytemuck::{Pod, Zeroable};

/// The side of the tile texture in pixels.
pub(crate) const TILE: u32 = 2048;

/// A rectangle of pixels: x, y, width, height.
pub(crate) type Rect = (u32, u32, u32, u32);

/// Mirrors the `Uniform` of `proxy.wgsl`; a test holds the two layouts equal.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ProxyUniform {
    pub(crate) source_size: [f32; 2],
    pub(crate) proxy_size: [f32; 2],
    pub(crate) tile_origin: [f32; 2],
    pub(crate) _pad: [f32; 2],
}

/// A block of the proxy and the source pixels that are drawn for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Tile {
    /// The proxy pixels this tile serves.
    pub(crate) proxy: Rect,
    /// The source pixels the tile texture holds for them, from its texel
    /// (0, 0) on.
    pub(crate) source: Rect,
}

impl Tile {
    /// The part of the source the tile covers, normalised: the window of the
    /// pass that draws it.
    pub(crate) fn window(&self, source: (u32, u32)) -> [f32; 4] {
        let (x, y, w, h) = self.source;
        let (sw, sh) = (source.0 as f32, source.1 as f32);
        [x as f32 / sw, y as f32 / sh, w as f32 / sw, h as f32 / sh]
    }

    pub(crate) fn uniform(&self, source: (u32, u32), proxy: (u32, u32)) -> ProxyUniform {
        ProxyUniform {
            source_size: [source.0 as f32, source.1 as f32],
            proxy_size: [proxy.0 as f32, proxy.1 as f32],
            tile_origin: [self.source.0 as f32, self.source.1 as f32],
            _pad: [0.0; 2],
        }
    }
}

/// The blocks along one axis: (first proxy pixel, proxy pixels, first source
/// pixel, source pixels). The source span of a block is what `proxy.wgsl`
/// reads for its proxy pixels, with a pixel to spare past its end for the
/// rounding of the shader's own sum.
fn blocks(source: u32, proxy: u32) -> Vec<(u32, u32, u32, u32)> {
    let span = source as f32 / proxy as f32;
    let step = (((TILE - 2) as f32 / span).floor() as u32).max(1);
    (0..proxy)
        .step_by(step as usize)
        .map(|first| {
            let count = step.min(proxy - first);
            let from = (first as f32 * span).floor() as u32;
            let to = (((first + count) as f32 * span).ceil() as u32 + 1).min(source);
            (first, count, from, to - from)
        })
        .collect()
}

/// The tiles of a proxy of size `proxy` over a source of size `source`, row
/// by row.
pub(crate) fn tiles(source: (u32, u32), proxy: (u32, u32)) -> Vec<Tile> {
    let across = blocks(source.0, proxy.0);
    blocks(source.1, proxy.1)
        .into_iter()
        .flat_map(|(py, ph, sy, sh)| {
            across.iter().map(move |&(px, pw, sx, sw)| Tile {
                proxy: (px, py, pw, ph),
                source: (sx, sy, sw, sh),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gamut_color::brush::Proxy;

    #[test]
    fn a_source_that_fits_a_tile_is_one_tile_of_itself() {
        for size in [(64, 64), (800, 600), (1920, 1080), (2046, 100)] {
            let proxy = Proxy::size_for(size);
            assert_eq!(
                tiles(size, proxy),
                [Tile {
                    proxy: (0, 0, proxy.0, proxy.1),
                    source: (0, 0, size.0, size.1),
                }],
                "{size:?}"
            );
        }
    }

    #[test]
    fn the_tiles_serve_every_proxy_pixel_once_and_hold_every_source_pixel_it_reads() {
        for source in [
            (6000, 4000),
            (4000, 6000),
            (3840, 2160),
            (12000, 9000),
            (8192, 1),
        ] {
            let proxy = Proxy::size_for(source);
            let tiles = tiles(source, proxy);
            let mut served = vec![0u8; (proxy.0 * proxy.1) as usize];
            for tile in &tiles {
                let (px, py, pw, ph) = tile.proxy;
                let (sx, sy, sw, sh) = tile.source;
                assert!(sw <= TILE && sh <= TILE, "{source:?} {tile:?}");
                assert!(sx + sw <= source.0 && sy + sh <= source.1);
                for y in py..py + ph {
                    for x in px..px + pw {
                        served[(y * proxy.0 + x) as usize] += 1;
                    }
                }
                // What proxy.wgsl reads for the corner pixels of the block,
                // by its own arithmetic.
                for (first, count, from, held, size, reduced) in [
                    (px, pw, sx, sw, source.0, proxy.0),
                    (py, ph, sy, sh, source.1, proxy.1),
                ] {
                    let span = size as f32 / reduced as f32;
                    let low = (first as f32 * span).floor() as u32;
                    let last = first + count - 1;
                    let high = ((last as f32 * span + span).ceil() as u32).min(size);
                    assert!(low >= from && high <= from + held, "{source:?} {tile:?}");
                }
            }
            assert!(served.iter().all(|count| *count == 1), "{source:?}");
        }
        assert_eq!(tiles((6000, 4000), (1024, 683)).len(), 3 * 2);
        assert_eq!(tiles((3840, 2160), (1024, 576)).len(), 2 * 2);
    }

    #[test]
    fn a_tile_window_is_its_source_pixels_over_the_source() {
        let tile = Tile {
            proxy: (0, 0, 10, 10),
            source: (1500, 1000, 1500, 500),
        };
        assert_eq!(tile.window((6000, 4000)), [0.25, 0.25, 0.25, 0.125]);
        assert_eq!(
            tile.uniform((6000, 4000), (1024, 683)).tile_origin,
            [1500.0, 1000.0]
        );
    }
}
