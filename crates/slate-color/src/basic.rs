//! The Basic panel on the CPU: one function per operator on a linear
//! Rec.2020 pixel. Every function here is the reference the shader in
//! slate-gpu's develop.wgsl is tested against, so the two mirror each other
//! line for line. Nothing in this module clips; clipping happens in the
//! output transform.

use slate_core::PhotoEdit;

use crate::SourceSpace;
use crate::bradford;
use crate::daylight::{NEUTRAL_CCT, daylight_xy};
use crate::matrices::{self, Mat3, xyz_from_xy};
use crate::transfer::{linear_to_srgb8, srgb8_to_linear};

/// The Rec.2020 luminance weights.
pub const LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// Middle grey, the pivot of highlights, shadows and contrast.
pub const GREY: f32 = 0.18;

/// The stops of push at the ends of the highlights and shadows ranges.
pub const HIGHLIGHTS_SHADOWS_STOPS: f32 = 1.0;

/// Highlights reach full weight this many stops above grey.
pub const HIGHLIGHTS_RANGE: f32 = 3.0;

/// Shadows reach full weight this many stops below grey.
pub const SHADOWS_RANGE: f32 = 4.0;

/// Contrast pushes at most this many stops away from the pivot.
pub const CONTRAST_STOPS: f32 = 1.0;

/// The width in stops over which the contrast curve saturates.
pub const CONTRAST_WIDTH: f32 = 2.0;

/// The whites slider at 100 moves the clip point by this fraction.
pub const WHITES_RANGE: f32 = 0.25;

/// The blacks slider at 100 moves the black point by this fraction.
pub const BLACKS_RANGE: f32 = 0.05;

/// The tint slider at 100 moves the target white's y by this much.
pub const TINT_RANGE: f64 = 0.05;

/// The base layer blur sigma as a fraction of the short edge.
pub const BASE_SIGMA_FRACTION: f32 = 0.02;

/// The largest gaussian radius the blur runs, in pixels.
pub const MAX_BLUR_RADIUS: i32 = 255;

const EPSILON: f32 = 1e-6;

/// Rec.2020 luminance.
pub fn luma(px: [f32; 3]) -> f32 {
    px[0] * LUMA[0] + px[1] * LUMA[1] + px[2] * LUMA[2]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn scale(px: [f32; 3], gain: f32) -> [f32; 3] {
    px.map(|c| c * gain)
}

/// The Rec.2020 to Rec.2020 matrix for a temperature and tint setting.
///
/// The slider maps to a target white on the daylight locus at
/// 6504 K times 2 to the power of temperature over 100, with its y moved by
/// tint over 100 times [`TINT_RANGE`]. The Bradford adaptation always runs
/// from that target white to the locus white at 6504 K (within 0.00002 of
/// D65, and exactly the identity at zero). A positive temperature names a
/// bluer white, so mapping it to neutral warms the picture; a negative one
/// names a redder white, so the picture cools. A positive tint names a
/// greener white, so the picture shifts toward magenta.
pub fn white_balance_matrix(temperature: f32, tint: f32) -> Mat3 {
    if temperature == 0.0 && tint == 0.0 {
        return Mat3::IDENTITY;
    }
    let cct = NEUTRAL_CCT * 2f64.powf(f64::from(temperature) / 100.0);
    let (x, y) = daylight_xy(cct);
    let y = y + f64::from(tint) / 100.0 * TINT_RANGE;
    let (nx, ny) = daylight_xy(NEUTRAL_CCT);
    let adapt = bradford::adaptation(xyz_from_xy(x, y), xyz_from_xy(nx, ny));
    matrices::xyz_to_rec2020() * adapt * matrices::REC2020_TO_XYZ
}

/// White balance: see [`white_balance_matrix`].
pub fn white_balance(px: [f32; 3], temperature: f32, tint: f32) -> [f32; 3] {
    white_balance_matrix(temperature, tint).apply(px)
}

/// Exposure: 2 to the power of `ev`.
pub fn exposure(px: [f32; 3], ev: f32) -> [f32; 3] {
    scale(px, 2f32.powf(ev))
}

/// Highlights and shadows on the two-layer split. `base_luma` is the blurred
/// luminance under this pixel, already scaled by the exposure. The base is
/// pushed by up to [`HIGHLIGHTS_SHADOWS_STOPS`] on a smooth weight that
/// starts at grey, and the whole pixel takes the same gain, so the detail
/// (the ratio of the pixel to its base) is unchanged. Negative highlights
/// darken the bright base, positive shadows lift the dark base, as in
/// Lightroom.
pub fn highlights_shadows(px: [f32; 3], base_luma: f32, highlights: f32, shadows: f32) -> [f32; 3] {
    let l = (base_luma.max(EPSILON) / GREY).log2();
    let highlight_weight = smoothstep(0.0, HIGHLIGHTS_RANGE, l);
    let shadow_weight = smoothstep(0.0, SHADOWS_RANGE, -l);
    let stops = HIGHLIGHTS_SHADOWS_STOPS
        * (highlights / 100.0 * highlight_weight + shadows / 100.0 * shadow_weight);
    scale(px, 2f32.powf(stops))
}

/// Whites and blacks move the two end points. Whites at 100 lowers the
/// level that maps to 1.0 by [`WHITES_RANGE`], at -100 raises it by the
/// same. Blacks at 100 lifts the level that maps to 0.0 below zero by
/// [`BLACKS_RANGE`], at -100 raises it above zero.
pub fn whites_blacks(px: [f32; 3], whites: f32, blacks: f32) -> [f32; 3] {
    let white = 1.0 - WHITES_RANGE * whites / 100.0;
    let black = -BLACKS_RANGE * blacks / 100.0;
    px.map(|c| (c - black) / (white - black))
}

/// Contrast: an S-curve around grey in log2 space, applied to the
/// luminance and carried to every channel as a gain, so hue and chroma
/// hold. The curve is l plus strength times tanh(l over width), which has
/// unit slope at zero strength and saturates [`CONTRAST_STOPS`] away.
pub fn contrast(px: [f32; 3], contrast: f32) -> [f32; 3] {
    let l = (luma(px).max(EPSILON) / GREY).log2();
    let push = contrast / 100.0 * CONTRAST_STOPS * (l / CONTRAST_WIDTH).tanh();
    scale(px, 2f32.powf(push))
}

/// Vibrance and saturation in luminance plus chroma. Chroma is the vector
/// from grey. Saturation scales it by 1 plus the slider over 100. Vibrance
/// scales it by 1 plus the slider over 100 times one minus the normalised
/// chroma, the chroma length over the luminance clipped to 1, so pixels
/// that are already saturated move less.
pub fn vibrance_saturation(px: [f32; 3], vibrance: f32, saturation: f32) -> [f32; 3] {
    let l = luma(px);
    let chroma = px.map(|c| c - l);
    let length = (chroma[0] * chroma[0] + chroma[1] * chroma[1] + chroma[2] * chroma[2]).sqrt();
    let normalised = (length / l.max(EPSILON)).clamp(0.0, 1.0);
    let gain = (1.0 + vibrance / 100.0 * (1.0 - normalised)) * (1.0 + saturation / 100.0);
    [
        l + chroma[0] * gain,
        l + chroma[1] * gain,
        l + chroma[2] * gain,
    ]
}

/// The whole Basic panel in the ruled order: white balance, exposure,
/// highlights and shadows, whites and blacks, contrast, vibrance and
/// saturation. `base_luma` is the blurred luminance of the input pixel
/// before any operator; the exposure is applied to it here.
pub fn develop_pixel(px: [f32; 3], base_luma: f32, edit: &PhotoEdit) -> [f32; 3] {
    let wb = white_balance_matrix(edit.white_balance_temperature, edit.white_balance_tint);
    develop_pixel_with(px, base_luma, edit, &wb)
}

/// [`develop_pixel`] with the white balance matrix computed once.
pub fn develop_pixel_with(px: [f32; 3], base_luma: f32, edit: &PhotoEdit, wb: &Mat3) -> [f32; 3] {
    let px = wb.apply(px);
    let px = exposure(px, edit.exposure);
    let base = base_luma * 2f32.powf(edit.exposure);
    let px = highlights_shadows(px, base, edit.highlights, edit.shadows);
    let px = whites_blacks(px, edit.whites, edit.blacks);
    let px = contrast(px, edit.contrast);
    vibrance_saturation(px, edit.vibrance, edit.saturation)
}

/// An 8-bit encoded pixel in `space` to linear Rec.2020: the sRGB curve,
/// then the space's matrix.
pub fn decode_rgb8(rgb: [u8; 3], space: SourceSpace) -> [f32; 3] {
    matrices::input_matrix(space).apply(rgb.map(srgb8_to_linear))
}

/// The output transform: linear Rec.2020 to sRGB, clipped to 0 and 1, then
/// the sRGB curve and 8 bits.
pub fn output_srgb8(px: [f32; 3]) -> [u8; 3] {
    matrices::rec2020_to_srgb().apply(px).map(linear_to_srgb8)
}

/// The blur sigma of the base layer for a render of this size.
pub fn base_sigma(width: u32, height: u32) -> f32 {
    (BASE_SIGMA_FRACTION * width.min(height) as f32).max(0.5)
}

/// The kernel radius for a sigma: three sigma, capped.
pub fn blur_radius(sigma: f32) -> i32 {
    ((3.0 * sigma).ceil() as i32).min(MAX_BLUR_RADIUS)
}

/// The base layer of a linear image: its luminance under a separable
/// gaussian of [`base_sigma`], clamped at the edges. The shader does the
/// same arithmetic in the same order.
pub fn base_layer(pixels: &[[f32; 3]], width: u32, height: u32) -> Vec<f32> {
    let sigma = base_sigma(width, height);
    let radius = blur_radius(sigma);
    let weights: Vec<f32> = (-radius..=radius)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let (w, h) = (width as i32, height as i32);
    let luma_image: Vec<f32> = pixels.iter().map(|px| luma(*px)).collect();
    let pass = |source: &[f32], dx: i32, dy: i32| -> Vec<f32> {
        let mut out = vec![0.0; source.len()];
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for (k, weight) in weights.iter().enumerate() {
                    let i = k as i32 - radius;
                    let sx = (x + i * dx).clamp(0, w - 1);
                    let sy = (y + i * dy).clamp(0, h - 1);
                    sum += weight * source[(sy * w + sx) as usize];
                    weight_sum += weight;
                }
                out[(y * w + x) as usize] = sum / weight_sum;
            }
        }
        out
    };
    let horizontal = pass(&luma_image, 1, 0);
    pass(&horizontal, 0, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLES: [[f32; 3]; 6] = [
        [0.18, 0.18, 0.18],
        [0.9, 0.1, 0.05],
        [0.02, 0.4, 0.1],
        [0.05, 0.05, 0.6],
        [0.001, 0.002, 0.001],
        [0.7, 0.65, 0.2],
    ];

    fn assert_same(a: [f32; 3], b: [f32; 3], tolerance: f32, what: &str) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tolerance, "{what}: {a:?} is not {b:?}");
        }
    }

    #[test]
    fn every_operator_is_the_identity_at_zero() {
        for px in SAMPLES {
            assert_same(white_balance(px, 0.0, 0.0), px, 0.0, "white balance");
            assert_same(exposure(px, 0.0), px, 0.0, "exposure");
            assert_same(highlights_shadows(px, 0.3, 0.0, 0.0), px, 0.0, "highlights");
            assert_same(highlights_shadows(px, 0.02, 0.0, 0.0), px, 0.0, "shadows");
            assert_same(whites_blacks(px, 0.0, 0.0), px, 0.0, "whites and blacks");
            assert_same(contrast(px, 0.0), px, 0.0, "contrast");
            assert_same(vibrance_saturation(px, 0.0, 0.0), px, 1e-7, "vibrance");
            assert_same(
                develop_pixel(px, 0.2, &PhotoEdit::default()),
                px,
                1e-7,
                "develop",
            );
        }
    }

    #[test]
    fn one_stop_of_exposure_doubles() {
        assert_same(
            exposure([0.1, 0.2, 0.3], 1.0),
            [0.2, 0.4, 0.6],
            1e-7,
            "one stop",
        );
        assert_same(
            exposure([0.4, 0.4, 0.4], -1.0),
            [0.2, 0.2, 0.2],
            1e-7,
            "minus one",
        );
    }

    #[test]
    fn positive_temperature_warms_and_positive_tint_is_magenta() {
        let grey = [0.5, 0.5, 0.5];
        let warm = white_balance(grey, 50.0, 0.0);
        assert!(warm[0] > warm[2], "warm {warm:?} has more red than blue");
        let cool = white_balance(grey, -50.0, 0.0);
        assert!(cool[2] > cool[0], "cool {cool:?} has more blue than red");
        let magenta = white_balance(grey, 0.0, 50.0);
        assert!(
            magenta[1] < magenta[0] && magenta[1] < magenta[2],
            "magenta {magenta:?} has the least green"
        );
    }

    #[test]
    fn negative_highlights_darken_bright_areas_and_leave_dark_ones() {
        let bright = [0.8, 0.8, 0.8];
        let out = highlights_shadows(bright, 0.8, -100.0, 0.0);
        assert!(out[0] < 0.8 && out[0] > 0.4, "{out:?}");
        let dark = [0.02, 0.02, 0.02];
        assert_same(
            highlights_shadows(dark, 0.02, -100.0, 0.0),
            dark,
            1e-7,
            "dark",
        );
        let lifted = highlights_shadows(dark, 0.02, 0.0, 100.0);
        assert!(lifted[0] > 0.02 && lifted[0] <= 0.04, "{lifted:?}");
    }

    #[test]
    fn whites_and_blacks_move_the_end_points() {
        assert_same(
            whites_blacks([0.75; 3], 100.0, 0.0),
            [1.0; 3],
            1e-6,
            "whites up",
        );
        assert_same(
            whites_blacks([1.25; 3], -100.0, 0.0),
            [1.0; 3],
            1e-6,
            "whites down",
        );
        assert_same(
            whites_blacks([0.05; 3], 0.0, -100.0),
            [0.0; 3],
            1e-6,
            "blacks crushed",
        );
        let lifted = whites_blacks([0.0; 3], 0.0, 100.0);
        assert!(lifted[0] > 0.04 && lifted[0] < 0.05, "{lifted:?}");
    }

    #[test]
    fn contrast_pivots_on_grey() {
        assert_same(contrast([GREY; 3], 100.0), [GREY; 3], 1e-6, "pivot");
        let bright = contrast([0.5; 3], 100.0);
        assert!(bright[0] > 0.5, "{bright:?}");
        let dark = contrast([0.05; 3], 100.0);
        assert!(dark[0] < 0.05, "{dark:?}");
        let flat = contrast([0.5; 3], -100.0);
        assert!(flat[0] < 0.5, "{flat:?}");
    }

    #[test]
    fn saturation_scales_chroma_and_vibrance_spares_saturated_colours() {
        let pastel = [0.5, 0.45, 0.4];
        let more = vibrance_saturation(pastel, 0.0, 100.0);
        assert!((luma(more) - luma(pastel)).abs() < 1e-6, "luma holds");
        assert!(more[0] - more[2] > pastel[0] - pastel[2]);
        let grey = vibrance_saturation(pastel, 0.0, -100.0);
        assert_same(grey, [luma(pastel); 3], 1e-6, "fully desaturated");
        let vivid = [0.9, 0.05, 0.05];
        assert_same(
            vibrance_saturation(vivid, 100.0, 0.0),
            vivid,
            1e-6,
            "vivid holds",
        );
        let lifted = vibrance_saturation(pastel, 100.0, 0.0);
        assert!(lifted[0] - lifted[2] > pastel[0] - pastel[2]);
    }

    #[test]
    fn the_base_layer_of_a_flat_image_is_flat() {
        let pixels = vec![[0.3, 0.3, 0.3]; 16 * 16];
        let base = base_layer(&pixels, 16, 16);
        for value in base {
            assert!((value - 0.3).abs() < 1e-5, "{value}");
        }
    }

    #[test]
    fn decode_and_output_round_trip() {
        for code in [[0, 0, 0], [255, 255, 255], [128, 64, 200], [10, 250, 3]] {
            let px = decode_rgb8(code, SourceSpace::Srgb);
            assert_eq!(output_srgb8(px), code);
        }
    }
}
