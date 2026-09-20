//! The colour wheels in ASC CDL form on an ACEScct pixel:
//! out = (in times slope plus offset) to the power, per channel. Shadows
//! drives the offset (lift), midtones the power (gamma), highlights the slope
//! (gain).
//!
//! A wheel is a point on the unit disc and a luminance slider. The disc
//! point becomes a per-channel tint on the hue basis of [`crate::hue`], with
//! zero mean across channels, so moving a wheel never changes the overall
//! brightness; the luminance slider moves all three channels together.
//!
//! Each wheel is weighted per pixel by the luminance of the ACEScct pixel on
//! the normalised axis, so the shadows wheel tints the shadows and leaves the
//! highlights alone: the offset fades out by [`SHADOWS_END`], the slope fades
//! in from [`HIGHLIGHTS_START`], and the power follows a bell around the
//! middle of the axis.

use slate_core::look::{Wheel, Wheels};

use crate::{acescct, basic, hue};

/// The offset at full deflection.
pub const OFFSET_RANGE: f32 = 0.033;

/// The slope runs from 1 minus this to 1 plus this.
pub const SLOPE_RANGE: f32 = 0.17;

/// The power runs from 2 to the minus this to 2 to the plus this.
pub const POWER_STOPS: f32 = 0.33;

/// The shadows weight is 1 at black and 0 from here on the normalised axis.
pub const SHADOWS_END: f32 = 0.66;

/// The highlights weight is 0 up to here and 1 at diffuse white.
pub const HIGHLIGHTS_START: f32 = 0.33;

/// Scales the basis so a disc point at the rim toward red tints by
/// (1, -0.5, -0.5): the square root of 3 over 2.
pub const TINT_SCALE: f32 = 1.224_744_9;

/// The CDL the shader takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cdl {
    pub slope: [f32; 3],
    pub offset: [f32; 3],
    pub power: [f32; 3],
}

impl Cdl {
    pub const IDENTITY: Cdl = Cdl {
        slope: [1.0; 3],
        offset: [0.0; 3],
        power: [1.0; 3],
    };

    pub fn new(wheels: &Wheels) -> Self {
        let shadows = deflection(&wheels.shadows);
        let midtones = deflection(&wheels.midtones);
        let highlights = deflection(&wheels.highlights);
        Cdl {
            slope: highlights.map(|d| 1.0 + SLOPE_RANGE * d),
            offset: shadows.map(|d| OFFSET_RANGE * d),
            power: midtones.map(|d| 2f32.powf(-POWER_STOPS * d)),
        }
    }
}

/// The tint of a disc point: zero mean across channels. A point outside the
/// unit disc is pulled back to the rim.
pub fn tint(x: f32, y: f32) -> [f32; 3] {
    let length = (x * x + y * y).sqrt();
    let pull = if length > 1.0 { 1.0 / length } else { 1.0 };
    hue::from_plane([x * pull * TINT_SCALE, y * pull * TINT_SCALE])
}

/// The per-channel deflection of a wheel in -1 to 1: its tint plus its
/// luminance.
pub fn deflection(wheel: &Wheel) -> [f32; 3] {
    let luminance = wheel.luminance / 100.0;
    tint(wheel.x, wheel.y).map(|t| (t + luminance).clamp(-1.0, 1.0))
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The position of an ACEScct pixel on the normalised axis by its luminance:
/// 0 at black, 1 at diffuse white, clamped.
pub fn tone(v: [f32; 3]) -> f32 {
    acescct::normalise(basic::luma(v)).clamp(0.0, 1.0)
}

/// The weights of the shadows, midtones and highlights wheels at `n` on the
/// normalised axis, in that order.
pub fn weights(n: f32) -> [f32; 3] {
    [
        1.0 - smoothstep(0.0, SHADOWS_END, n),
        smoothstep(0.0, 1.0, 1.0 - (2.0 * n - 1.0).abs()),
        smoothstep(HIGHLIGHTS_START, 1.0, n),
    ]
}

/// The CDL on one ACEScct pixel, each parameter weighted by the tone of the
/// pixel. The base of the power is clamped at 0.
pub fn apply(v: [f32; 3], cdl: &Cdl) -> [f32; 3] {
    let [shadows, midtones, highlights] = weights(tone(v));
    [0, 1, 2].map(|c| {
        let slope = 1.0 + highlights * (cdl.slope[c] - 1.0);
        let offset = shadows * cdl.offset[c];
        let power = 1.0 + midtones * (cdl.power[c] - 1.0);
        (v[c] * slope + offset).max(0.0).powf(power)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIXELS: [[f32; 3]; 4] = [
        [0.41, 0.41, 0.41],
        [0.5, 0.3, 0.2],
        [0.08, 0.1, 0.3],
        [0.6, 0.55, 0.52],
    ];

    fn mean(v: [f32; 3]) -> f32 {
        (v[0] + v[1] + v[2]) / 3.0
    }

    #[test]
    fn wheels_at_the_centre_are_the_identity() {
        let cdl = Cdl::new(&Wheels::default());
        assert_eq!(cdl, Cdl::IDENTITY);
        for v in PIXELS {
            assert_eq!(apply(v, &cdl), v);
        }
    }

    #[test]
    fn a_deflected_wheel_has_zero_mean_across_channels() {
        for (x, y) in [(1.0, 0.0), (-0.3, 0.8), (0.5, -0.5), (3.0, 4.0)] {
            let t = tint(x, y);
            assert!(mean(t).abs() < 1e-6, "{t:?}");
            assert!(t.iter().all(|c| c.abs() <= 1.0 + 1e-6));
        }
        let red = tint(1.0, 0.0);
        assert!((red[0] - 1.0).abs() < 1e-6 && (red[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_deflected_shadow_wheel_leaves_the_channel_mean_unchanged() {
        let wheels = Wheels {
            shadows: Wheel {
                x: -0.4,
                y: -0.5,
                luminance: 0.0,
            },
            ..Wheels::default()
        };
        let cdl = Cdl::new(&wheels);
        for v in PIXELS {
            let out = apply(v, &cdl);
            assert!((mean(out) - mean(v)).abs() < 1e-6, "{v:?} to {out:?}");
            if tone(v) < SHADOWS_END {
                assert_ne!(out, v);
            }
        }
        assert!(PIXELS.iter().any(|v| tone(*v) < SHADOWS_END));
    }

    #[test]
    fn a_deflected_highlight_wheel_leaves_the_mean_of_a_grey_unchanged() {
        let wheels = Wheels {
            highlights: Wheel {
                x: 0.6,
                y: 0.3,
                luminance: 0.0,
            },
            ..Wheels::default()
        };
        let v = PIXELS[0];
        let out = apply(v, &Cdl::new(&wheels));
        assert!((mean(out) - mean(v)).abs() < 1e-6);
        assert!(out[0] > v[0]);
    }

    #[test]
    fn luminance_moves_all_three_channels_together() {
        let lift = Cdl::new(&Wheels {
            shadows: Wheel {
                luminance: 50.0,
                ..Wheel::default()
            },
            ..Wheels::default()
        });
        assert_eq!(lift.offset, [OFFSET_RANGE * 0.5; 3]);
        let gamma = Cdl::new(&Wheels {
            midtones: Wheel {
                luminance: 100.0,
                ..Wheel::default()
            },
            ..Wheels::default()
        });
        assert_eq!(gamma.power, [2f32.powf(-POWER_STOPS); 3]);
        let out = apply([0.4; 3], &gamma);
        assert!(out[0] > 0.4);
    }

    #[test]
    fn the_ranges_are_a_third_of_the_pure_cdl_ranges() {
        assert_eq!(OFFSET_RANGE, 0.033);
        assert_eq!(SLOPE_RANGE, 0.17);
        assert_eq!(POWER_STOPS, 0.33);
        assert_eq!(SHADOWS_END, 0.66);
        assert_eq!(HIGHLIGHTS_START, 0.33);
    }

    #[test]
    fn the_three_weights_stay_within_0_and_1() {
        for step in 0..=1000 {
            let n = step as f32 / 1000.0;
            for w in weights(n) {
                assert!((0.0..=1.0).contains(&w), "weight {w} at {n}");
            }
        }
    }

    #[test]
    fn the_shadows_weight_is_1_at_black_and_0_from_its_end() {
        assert_eq!(weights(0.0)[0], 1.0);
        for n in [SHADOWS_END, 0.7, 0.9, 1.0] {
            assert_eq!(weights(n)[0], 0.0, "at {n}");
        }
        assert!(weights(0.3)[0] > 0.0 && weights(0.3)[0] < 1.0);
    }

    #[test]
    fn the_highlights_weight_is_0_up_to_its_start_and_1_at_white() {
        for n in [0.0, 0.2, HIGHLIGHTS_START] {
            assert_eq!(weights(n)[2], 0.0, "at {n}");
        }
        assert_eq!(weights(1.0)[2], 1.0);
    }

    #[test]
    fn the_midtones_weight_is_a_bell_around_the_middle() {
        assert_eq!(weights(0.5)[1], 1.0);
        assert_eq!(weights(0.0)[1], 0.0);
        assert_eq!(weights(1.0)[1], 0.0);
        assert!((weights(0.25)[1] - weights(0.75)[1]).abs() < 1e-6);
    }

    #[test]
    fn the_tone_of_a_pixel_is_clamped_to_the_axis() {
        assert_eq!(tone([0.0; 3]), 0.0);
        assert_eq!(tone([2.0; 3]), 1.0);
        let middle = acescct::denormalise(0.5);
        assert!((tone([middle; 3]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_deflected_shadows_wheel_leaves_a_highlight_pixel_unchanged() {
        let cdl = Cdl::new(&Wheels {
            shadows: Wheel {
                x: -0.5,
                y: -0.3,
                luminance: 40.0,
            },
            ..Wheels::default()
        });
        let level = acescct::denormalise(0.9);
        let v = [level + 0.02, level, level - 0.02];
        assert!((tone(v) - 0.9).abs() < 0.01);
        let out = apply(v, &cdl);
        for c in 0..3 {
            assert!((out[c] - v[c]).abs() < 1e-6, "{v:?} to {out:?}");
        }
        let dark = [acescct::denormalise(0.1); 3];
        assert_ne!(apply(dark, &cdl), dark);
    }
}
