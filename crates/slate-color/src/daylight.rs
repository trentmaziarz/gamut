//! The CIE daylight locus: the chromaticity of daylight at a correlated
//! colour temperature, from the standard polynomial fit.

/// The correlated colour temperature the white balance slider is neutral at.
pub const NEUTRAL_CCT: f64 = 6504.0;

/// The xy chromaticity of daylight at `cct` kelvin. The fit is published for
/// 4000 K to 25000 K; below 4000 K the lower branch is extrapolated, which
/// is smooth and monotonic down to the 3252 K the slider reaches.
pub fn daylight_xy(cct: f64) -> (f64, f64) {
    let t = cct.clamp(2500.0, 25000.0);
    let t2 = t * t;
    let t3 = t2 * t;
    let x = if t <= 7000.0 {
        -4.6070e9 / t3 + 2.9678e6 / t2 + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t3 + 1.9018e6 / t2 + 0.24748e3 / t + 0.237040
    };
    let y = -3.000 * x * x + 2.870 * x - 0.275;
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::{NEUTRAL_CCT, daylight_xy};

    #[test]
    fn the_neutral_point_is_d65() {
        let (x, y) = daylight_xy(NEUTRAL_CCT);
        assert!((x - 0.31272).abs() < 1e-3, "x {x}");
        assert!((y - 0.32903).abs() < 1e-3, "y {y}");
    }

    #[test]
    fn warmer_temperatures_have_larger_x() {
        let mut last = daylight_xy(13008.0).0;
        for cct in (3252..=13008).rev().step_by(250) {
            let (x, _) = daylight_xy(f64::from(cct));
            assert!(x >= last, "x fell at {cct} K");
            last = x;
        }
    }

    #[test]
    fn the_two_branches_meet_at_7000_k() {
        let (a, _) = daylight_xy(6999.9);
        let (b, _) = daylight_xy(7000.1);
        assert!((a - b).abs() < 1e-4);
    }
}
