//! Texture and clarity: local contrast at two scales. Each raises the ratio
//! of a pixel's luminance to a blurred luminance under it to a power, and
//! the whole pixel takes that gain, so hue and chroma hold.
//!
//! Clarity uses the base layer of highlights and shadows (sigma 2 percent of
//! the short edge) under a midtone bell. Texture uses a second, finer blur
//! (sigma 0.25 percent of the short edge) with no bell.
//!
//! The ratio is taken between the input pixel and the input base, before any
//! operator. With the other sliders at rest that equals the developed pixel
//! over the base times 2 to the EV; with them moved it still reads 1 on a
//! flat field, where white balance or contrast would otherwise turn a
//! luminance change into false detail.

use crate::basic::{GREY, gaussian, luma};

/// The exponent at a slider value of 100.
pub const STRENGTH: f32 = 0.5;

/// The half width of the clarity bell in stops around grey.
pub const CLARITY_HALF_WIDTH: f32 = 3.0;

/// The texture blur sigma as a fraction of the short edge.
pub const TEXTURE_SIGMA_FRACTION: f32 = 0.0025;

/// The smallest texture sigma in pixels.
pub const TEXTURE_SIGMA_FLOOR: f32 = 1.0;

const EPSILON: f32 = 1e-6;

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The blur sigma of the texture layer for a render of this size.
pub fn texture_sigma(width: u32, height: u32) -> f32 {
    (TEXTURE_SIGMA_FRACTION * width.min(height) as f32).max(TEXTURE_SIGMA_FLOOR)
}

/// The texture layer of a linear image: its luminance under a gaussian of
/// [`texture_sigma`].
pub fn texture_layer(pixels: &[[f32; 3]], width: u32, height: u32) -> Vec<f32> {
    let luma_image: Vec<f32> = pixels.iter().map(|px| luma(*px)).collect();
    gaussian(&luma_image, width, height, texture_sigma(width, height))
}

/// The weight of clarity at a base luminance: 1 at grey, 0 from
/// [`CLARITY_HALF_WIDTH`] stops away.
pub fn midtone_bell(exposed_base: f32) -> f32 {
    let l = (exposed_base.max(EPSILON) / GREY).log2();
    1.0 - smoothstep(0.0, CLARITY_HALF_WIDTH, l.abs())
}

/// The gain texture and clarity put on a pixel. `input_luma` is the
/// luminance of the input pixel, `base` and `texture_base` the two blurred
/// luminances under it, and `exposed_base` the base times 2 to the EV.
pub fn gain(
    input_luma: f32,
    base: f32,
    texture_base: f32,
    exposed_base: f32,
    texture: f32,
    clarity: f32,
) -> f32 {
    let l = input_luma.max(EPSILON);
    let clarity_power = clarity / 100.0 * STRENGTH * midtone_bell(exposed_base);
    let texture_power = texture / 100.0 * STRENGTH;
    (l / base.max(EPSILON)).powf(clarity_power)
        * (l / texture_base.max(EPSILON)).powf(texture_power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_field_is_unchanged() {
        for level in [0.02, 0.18, 0.7] {
            let g = gain(level, level, level, level, 100.0, 100.0);
            assert!((g - 1.0).abs() < 1e-6, "{level}: {g}");
            let g = gain(level, level, level, level * 2.0, -100.0, -100.0);
            assert!((g - 1.0).abs() < 1e-6, "{level}: {g}");
        }
    }

    #[test]
    fn zero_sliders_are_the_identity_on_any_detail() {
        assert_eq!(gain(0.3, 0.2, 0.25, 0.2, 0.0, 0.0), 1.0);
    }

    #[test]
    fn positive_clarity_pushes_a_pixel_away_from_its_base() {
        assert!(gain(0.3, 0.2, 0.3, 0.2, 0.0, 50.0) > 1.0);
        assert!(gain(0.1, 0.2, 0.1, 0.2, 0.0, 50.0) < 1.0);
        assert!(gain(0.3, 0.2, 0.3, 0.2, 0.0, -50.0) < 1.0);
    }

    #[test]
    fn clarity_fades_out_three_stops_from_grey_and_texture_does_not() {
        assert_eq!(midtone_bell(GREY), 1.0);
        assert!(midtone_bell(GREY * 8.0) < 1e-5);
        assert!(midtone_bell(GREY / 8.0) < 1e-5);
        assert!((gain(2.0, 1.6, 2.0, 1.6, 0.0, 100.0) - 1.0).abs() < 1e-6);
        assert!(gain(2.0, 2.0, 1.6, 1.6, 100.0, 0.0) > 1.0);
    }

    #[test]
    fn the_texture_sigma_has_a_floor_of_one_pixel() {
        assert_eq!(texture_sigma(64, 64), 1.0);
        assert_eq!(texture_sigma(4000, 6000), 10.0);
    }
}
