//! The colour wheels in ASC CDL form on an ACEScct pixel:
//! out = (in times slope plus offset) to the power, per channel. Shadows
//! drives the offset (lift), midtones the power (gamma), highlights the slope
//! (gain).
//!
//! A wheel is a point on the unit disc and a luminance slider. The disc
//! point becomes a per-channel tint on the hue basis of [`crate::hue`], with
//! zero mean across channels, so moving a wheel never changes the overall
//! brightness; the luminance slider moves all three channels together.

use slate_core::look::{Wheel, Wheels};

use crate::hue;

/// The offset at full deflection.
pub const OFFSET_RANGE: f32 = 0.1;

/// The slope runs from 1 minus this to 1 plus this.
pub const SLOPE_RANGE: f32 = 0.5;

/// The power runs from 2 to the minus this to 2 to the plus this.
pub const POWER_STOPS: f32 = 1.0;

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

/// The CDL on one ACEScct pixel. The base of the power is clamped at 0.
pub fn apply(v: [f32; 3], cdl: &Cdl) -> [f32; 3] {
    [0, 1, 2].map(|c| {
        (v[c] * cdl.slope[c] + cdl.offset[c])
            .max(0.0)
            .powf(cdl.power[c])
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
            assert_ne!(out, v);
        }
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
        assert_eq!(lift.offset, [0.05; 3]);
        let gamma = Cdl::new(&Wheels {
            midtones: Wheel {
                luminance: 100.0,
                ..Wheel::default()
            },
            ..Wheels::default()
        });
        assert_eq!(gamma.power, [0.5; 3]);
        let out = apply([0.4; 3], &gamma);
        assert!(out[0] > 0.4);
    }
}
