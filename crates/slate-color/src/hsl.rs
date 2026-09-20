//! The HSL mixer on an ACEScct pixel: eight hue ranges, each with a hue
//! shift, a chroma scale and a luminance offset, blended by the raised
//! cosine weights of [`crate::hue`].

use slate_core::look::{HSL_RANGES, HslRange};

use crate::acescct::LOG_SCALE;
use crate::basic::luma;
use crate::hue;

/// The hue slider at 100 turns the hue by this many degrees.
pub const HUE_DEGREES: f32 = 30.0;

/// The luminance slider at 100 moves the pixel by this many stops.
pub const LUMINANCE_STOPS: f32 = 1.0;

/// Pixels with less chroma than this are left alone, so noise in greys does
/// not pick up colour.
pub const CHROMA_FLOOR: f32 = 0.01;

/// The mixer reaches full strength at this chroma. The ramp from
/// [`CHROMA_FLOOR`] is kept thin: an overcast photo has most of its pixels
/// under a chroma of 0.02, and a wide ramp leaves the mixer nothing to move
/// there. It exists so the floor is not a step the CPU and the GPU could
/// land on opposite sides of.
pub const CHROMA_FULL: f32 = 0.012;

/// The mixer as the shader takes it: per range the hue turn in radians, the
/// chroma scale, the luminance offset in ACEScct units, and a spare.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HslParams {
    pub ranges: [[f32; 4]; HSL_RANGES],
    pub centres: [f32; HSL_RANGES],
}

impl HslParams {
    pub fn new(ranges: &[HslRange; HSL_RANGES]) -> Self {
        HslParams {
            ranges: ranges.map(|r| {
                [
                    r.hue / 100.0 * HUE_DEGREES.to_radians(),
                    1.0 + r.saturation / 100.0,
                    r.luminance / 100.0 * LUMINANCE_STOPS / LOG_SCALE,
                    0.0,
                ]
            }),
            centres: hue::range_centres(),
        }
    }
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The mixer on one ACEScct pixel. The chroma vector is turned and scaled on
/// the hue plane, and the pixel is rebuilt around its own luminance plus the
/// luminance offset, so a hue or saturation move leaves the luminance alone.
pub fn apply(v: [f32; 3], params: &HslParams) -> [f32; 3] {
    let plane = hue::chroma_plane(v);
    let chroma = hue::chroma(plane);
    let strength = smoothstep(CHROMA_FLOOR, CHROMA_FULL, chroma);
    if strength <= 0.0 {
        return v;
    }
    let (lower, upper, weight) = hue::range_pair(&params.centres, hue::hue(plane));
    let (a, b) = (params.ranges[lower], params.ranges[upper]);
    let mix = |i: usize| a[i] * (1.0 - weight) + b[i] * weight;
    let turn = mix(0) * strength;
    let scale = 1.0 + (mix(1) - 1.0) * strength;
    let offset = mix(2) * strength;

    let (sin, cos) = turn.sin_cos();
    let turned = [
        (plane[0] * cos - plane[1] * sin) * scale,
        (plane[0] * sin + plane[1] * cos) * scale,
    ];
    let coloured = hue::from_plane(turned);
    let grey = luma(v) - luma(coloured) + offset;
    coloured.map(|c| c + grey)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SourceSpace;
    use crate::acescct::encode_pixel;
    use crate::basic::decode_rgb8;

    fn encoded(rgb: [u8; 3]) -> [f32; 3] {
        encode_pixel(decode_rgb8(rgb, SourceSpace::Srgb))
    }

    fn only(index: usize, range: HslRange) -> HslParams {
        let mut ranges = [HslRange::default(); HSL_RANGES];
        ranges[index] = range;
        HslParams::new(&ranges)
    }

    #[test]
    fn the_default_mixer_is_the_identity() {
        let params = HslParams::new(&[HslRange::default(); HSL_RANGES]);
        for rgb in [
            [200, 40, 30],
            [20, 90, 200],
            [128, 128, 128],
            [250, 240, 20],
        ] {
            let v = encoded(rgb);
            let out = apply(v, &params);
            for (a, b) in v.iter().zip(out) {
                assert!((a - b).abs() < 1e-6, "{v:?} became {out:?}");
            }
        }
    }

    #[test]
    fn orange_saturation_moves_orange_and_leaves_blue_and_grey() {
        let params = only(
            1,
            HslRange {
                saturation: -60.0,
                ..HslRange::default()
            },
        );
        let orange = encoded([230, 120, 20]);
        let out = apply(orange, &params);
        let before = hue::chroma(hue::chroma_plane(orange));
        let after = hue::chroma(hue::chroma_plane(out));
        assert!(after < before * 0.6, "{before} to {after}");
        assert!((luma(out) - luma(orange)).abs() < 1e-6);

        for rgb in [[20, 40, 220], [128, 128, 128], [130, 128, 127]] {
            let v = encoded(rgb);
            let out = apply(v, &params);
            for (a, b) in v.iter().zip(out) {
                assert!((a - b).abs() < 1e-6, "{v:?} became {out:?}");
            }
        }
    }

    #[test]
    fn hue_turns_the_angle_and_luminance_moves_one_stop() {
        let centres = hue::range_centres();
        let green = encoded([0, 255, 0]);
        let turned = apply(
            green,
            &only(
                3,
                HslRange {
                    hue: 100.0,
                    ..HslRange::default()
                },
            ),
        );
        let angle = hue::hue(hue::chroma_plane(turned));
        assert!((angle - centres[3] - HUE_DEGREES.to_radians()).abs() < 1e-4);

        let lifted = apply(
            green,
            &only(
                3,
                HslRange {
                    luminance: 100.0,
                    ..HslRange::default()
                },
            ),
        );
        assert!((luma(lifted) - luma(green) - 1.0 / LOG_SCALE).abs() < 1e-6);
    }
}
