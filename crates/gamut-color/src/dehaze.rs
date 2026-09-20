//! Dehaze by the dark channel prior (He, Sun and Tang 2009). The haze model
//! is I = J t + A (1 - t): the picture I is the scene J seen through a
//! transmission t, plus the atmospheric light A scattered in.
//!
//! The atmospheric light is a property of the source and is estimated once
//! on the CPU from a small downsample. The transmission map is the dark
//! channel of I over A under a square minimum filter, t = 1 - 0.95 times
//! that. The paper refines the map with a guided filter; Gamut smooths it
//! with one gaussian at the patch size instead, which is cheap on the GPU
//! and hides the patch edges well enough for a slider.

use crate::SourceSpace;
use crate::basic::gaussian_stored;
use crate::matrices;
use crate::transfer::srgb8_to_linear;

/// The patch of the minimum filter as a fraction of the short edge.
pub const PATCH_FRACTION: f32 = 0.015;

/// How much of the haze the map accounts for; the paper's omega.
pub const OMEGA: f32 = 0.95;

/// The transmission map never falls below this.
pub const TRANSMISSION_FLOOR: f32 = 0.01;

/// The divisor of the recovery never falls below this; the paper's t0.
pub const RECOVERY_FLOOR: f32 = 0.1;

/// The atmospheric light is the mean colour of this fraction of the
/// downsample, the brightest of its dark channel.
pub const BRIGHTEST_FRACTION: f32 = 0.001;

/// The long edge of the downsample the atmospheric light is read from.
pub const ATMOSPHERE_EDGE: u32 = 256;

/// No channel of the atmospheric light falls below this.
pub const ATMOSPHERE_FLOOR: f32 = 0.05;

/// The atmospheric light of video frames in M3: white.
pub const WHITE_ATMOSPHERE: [f32; 3] = [1.0; 3];

/// The radius of the minimum filter for a render of this size, in pixels.
pub fn patch_radius(width: u32, height: u32) -> i32 {
    ((PATCH_FRACTION * width.min(height) as f32 / 2.0).ceil() as i32).max(1)
}

/// The sigma of the gaussian that smooths the map: the patch size.
pub fn smoothing_sigma(width: u32, height: u32) -> f32 {
    (PATCH_FRACTION * width.min(height) as f32).max(0.5)
}

/// The smallest channel of a pixel over the atmospheric light.
pub fn dark(px: [f32; 3], atmosphere: [f32; 3]) -> f32 {
    (px[0] / atmosphere[0])
        .min(px[1] / atmosphere[1])
        .min(px[2] / atmosphere[2])
}

/// A square minimum filter, run as a row pass and a column pass, clamped at
/// the edges.
pub fn minimum(source: &[f32], width: u32, height: u32, radius: i32) -> Vec<f32> {
    minimum_stored(source, width, height, radius, &|v| v, &|v| v)
}

/// [`minimum`] with `store_rows` applied to what the row pass writes and
/// `store_columns` to what the column pass writes, for the golden tests.
pub fn minimum_stored(
    source: &[f32],
    width: u32,
    height: u32,
    radius: i32,
    store_rows: &dyn Fn(f32) -> f32,
    store_columns: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    let (w, h) = (width as i32, height as i32);
    let pass = |source: &[f32], dx: i32, dy: i32, store: &dyn Fn(f32) -> f32| -> Vec<f32> {
        let mut out = vec![0.0; source.len()];
        for y in 0..h {
            for x in 0..w {
                let mut least = f32::MAX;
                for i in -radius..=radius {
                    let sx = (x + i * dx).clamp(0, w - 1);
                    let sy = (y + i * dy).clamp(0, h - 1);
                    least = least.min(source[(sy * w + sx) as usize]);
                }
                out[(y * w + x) as usize] = store(least);
            }
        }
        out
    };
    let rows = pass(source, 1, 0, store_rows);
    pass(&rows, 0, 1, store_columns)
}

/// The transmission of a dark channel value.
pub fn transmission_of(dark: f32) -> f32 {
    (1.0 - OMEGA * dark).clamp(TRANSMISSION_FLOOR, 1.0)
}

/// The transmission map of a linear image.
pub fn transmission(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    atmosphere: [f32; 3],
) -> Vec<f32> {
    transmission_stored(pixels, width, height, atmosphere, &|v| v)
}

/// [`transmission`] with `store` applied wherever the GPU writes a texture:
/// the row minimum, the transmission of the column minimum, and the two
/// passes of the smoothing.
pub fn transmission_stored(
    pixels: &[[f32; 3]],
    width: u32,
    height: u32,
    atmosphere: [f32; 3],
    store: &dyn Fn(f32) -> f32,
) -> Vec<f32> {
    let dark_channel: Vec<f32> = pixels.iter().map(|px| dark(*px, atmosphere)).collect();
    let raw = minimum_stored(
        &dark_channel,
        width,
        height,
        patch_radius(width, height),
        store,
        &|least| store(transmission_of(least)),
    );
    gaussian_stored(&raw, width, height, smoothing_sigma(width, height), store)
}

/// A linear image box-averaged down to at most [`ATMOSPHERE_EDGE`] on its
/// long edge, with its new size. `pixel` reads one source pixel.
fn downsample(
    width: u32,
    height: u32,
    pixel: impl Fn(usize) -> [f32; 3],
) -> (Vec<[f32; 3]>, u32, u32) {
    let factor = width.max(height).div_ceil(ATMOSPHERE_EDGE).max(1);
    let (small_w, small_h) = (width.div_ceil(factor), height.div_ceil(factor));
    let mut small = Vec::with_capacity((small_w * small_h) as usize);
    for sy in 0..small_h {
        for sx in 0..small_w {
            let mut sum = [0.0f32; 3];
            let mut count = 0.0;
            for y in sy * factor..((sy + 1) * factor).min(height) {
                for x in sx * factor..((sx + 1) * factor).min(width) {
                    let px = pixel((y * width + x) as usize);
                    sum = [sum[0] + px[0], sum[1] + px[1], sum[2] + px[2]];
                    count += 1.0;
                }
            }
            small.push(sum.map(|c| c / count));
        }
    }
    (small, small_w, small_h)
}

/// The atmospheric light of a linear image: the image is box-averaged down
/// to at most [`ATMOSPHERE_EDGE`] on its long edge, and the answer is the
/// mean colour of the brightest [`BRIGHTEST_FRACTION`] of the dark channel.
pub fn atmosphere(pixels: &[[f32; 3]], width: u32, height: u32) -> [f32; 3] {
    if pixels.is_empty() {
        return WHITE_ATMOSPHERE;
    }
    let (small, small_w, small_h) = downsample(width, height, |i| pixels[i]);
    atmosphere_of_downsample(&small, small_w, small_h)
}

/// [`atmosphere`] of an 8-bit RGBA photo in `space`. The sRGB curve is
/// decoded through a table and the box average is taken before the matrix,
/// which is linear, so a 24 megapixel photo costs one table read per byte.
pub fn atmosphere_rgba8(rgba8: &[u8], width: u32, height: u32, space: SourceSpace) -> [f32; 3] {
    if rgba8.len() < (width as usize) * (height as usize) * 4 || width == 0 || height == 0 {
        return WHITE_ATMOSPHERE;
    }
    let table: Vec<f32> = (0..=255u8).map(srgb8_to_linear).collect();
    let (small, small_w, small_h) = downsample(width, height, |i| {
        [
            table[rgba8[i * 4] as usize],
            table[rgba8[i * 4 + 1] as usize],
            table[rgba8[i * 4 + 2] as usize],
        ]
    });
    let matrix = matrices::input_matrix(space);
    let small: Vec<[f32; 3]> = small.into_iter().map(|px| matrix.apply(px)).collect();
    atmosphere_of_downsample(&small, small_w, small_h)
}

fn atmosphere_of_downsample(small: &[[f32; 3]], small_w: u32, small_h: u32) -> [f32; 3] {
    let dark_channel: Vec<f32> = small.iter().map(|px| dark(*px, WHITE_ATMOSPHERE)).collect();
    let filtered = minimum(
        &dark_channel,
        small_w,
        small_h,
        patch_radius(small_w, small_h),
    );
    let mut order: Vec<usize> = (0..filtered.len()).collect();
    order.sort_by(|a, b| filtered[*b].total_cmp(&filtered[*a]));
    let keep = ((filtered.len() as f32 * BRIGHTEST_FRACTION).ceil() as usize).max(1);
    let mut sum = [0.0f32; 3];
    for &i in &order[..keep] {
        sum = [
            sum[0] + small[i][0],
            sum[1] + small[i][1],
            sum[2] + small[i][2],
        ];
    }
    sum.map(|c| (c / keep as f32).max(ATMOSPHERE_FLOOR))
}

/// The scene under the haze: J = (I - A) / max(t to the power s, 0.1) + A
/// with s the slider over 100. A negative slider adds haze through the same
/// formula.
pub fn recover(px: [f32; 3], transmission: f32, atmosphere: [f32; 3], dehaze: f32) -> [f32; 3] {
    let divisor = transmission
        .max(TRANSMISSION_FLOOR)
        .powf(dehaze / 100.0)
        .max(RECOVERY_FLOOR);
    [0, 1, 2].map(|c| (px[c] - atmosphere[c]) / divisor + atmosphere[c])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_transmission_or_a_zero_slider_is_the_identity() {
        let a = [0.8, 0.85, 0.9];
        for px in [[0.2, 0.3, 0.4], [0.9, 0.1, 0.0], [1.2, 1.0, 0.7]] {
            for out in [
                recover(px, 1.0, a, 100.0),
                recover(px, 1.0, a, -100.0),
                recover(px, 0.4, a, 0.0),
            ] {
                for (x, y) in out.iter().zip(px) {
                    assert!((x - y).abs() < 1e-6, "{px:?} became {out:?}");
                }
            }
        }
    }

    #[test]
    fn the_haze_model_inverts() {
        let a = [0.8, 0.85, 0.9];
        let scene = [0.1, 0.3, 0.05];
        let t = 0.5;
        let hazy = [0, 1, 2].map(|c| scene[c] * t + a[c] * (1.0 - t));
        let back = recover(hazy, t, a, 100.0);
        for (x, y) in back.iter().zip(scene) {
            assert!((x - y).abs() < 1e-6, "{back:?} is not {scene:?}");
        }
        let hazier = recover(scene, t, a, -100.0);
        assert!(hazier[0] > scene[0] && hazier[0] < a[0]);
    }

    #[test]
    fn the_minimum_filter_spreads_the_darkest_value_over_its_patch() {
        let mut image = vec![1.0; 7 * 7];
        image[3 * 7 + 3] = 0.2;
        let out = minimum(&image, 7, 7, 1);
        for y in 0..7 {
            for x in 0..7 {
                let inside = (2..=4).contains(&x) && (2..=4).contains(&y);
                assert_eq!(out[y * 7 + x], if inside { 0.2 } else { 1.0 });
            }
        }
    }

    #[test]
    fn a_hazy_image_gets_a_bright_atmosphere_and_a_low_transmission() {
        let (w, h) = (32u32, 32u32);
        let pixels: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let scene = if (i / w + i % w) % 2 == 0 { 0.05 } else { 0.4 };
                let t = if i / w < 16 { 0.3 } else { 1.0 };
                [scene * t + 0.9 * (1.0 - t); 3]
            })
            .collect();
        let a = atmosphere(&pixels, w, h);
        assert!(a[0] > 0.6 && a[0] < 1.0, "{a:?}");
        let map = transmission(&pixels, w, h, a);
        let top = map[(4 * w + 16) as usize];
        let bottom = map[(28 * w + 16) as usize];
        assert!(
            top < bottom,
            "haze at the top {top}, clear at the bottom {bottom}"
        );
        assert!(map.iter().all(|t| (TRANSMISSION_FLOOR..=1.0).contains(t)));
    }

    #[test]
    fn the_patch_scales_with_the_short_edge() {
        assert_eq!(patch_radius(64, 64), 1);
        assert_eq!(patch_radius(1280, 1600), 10);
        assert!((smoothing_sigma(1280, 1600) - 19.2).abs() < 1e-4);
    }
}
