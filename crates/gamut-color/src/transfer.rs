//! Transfer functions. M1 needs the sRGB pair: the EOTF that takes an
//! encoded value to linear light and the OETF that takes it back.

/// Encoded sRGB in 0 to 1 to linear light.
pub fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light in 0 to 1 to encoded sRGB.
pub fn srgb_oetf(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// An 8-bit sRGB code to linear light.
pub fn srgb8_to_linear(code: u8) -> f32 {
    srgb_eotf(f32::from(code) / 255.0)
}

/// Linear light to the nearest 8-bit sRGB code, clipping to 0 and 1 first.
pub fn linear_to_srgb8(v: f32) -> u8 {
    (srgb_oetf(v.clamp(0.0, 1.0)) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_srgb_curve_round_trips() {
        for i in 0..=1000 {
            let v = i as f32 / 1000.0;
            let back = srgb_oetf(srgb_eotf(v));
            assert!((back - v).abs() < 1e-6, "{v} came back as {back}");
        }
    }

    #[test]
    fn known_points() {
        assert!((srgb_eotf(1.0) - 1.0).abs() < 1e-7);
        assert!((srgb_eotf(0.5) - 0.214041).abs() < 1e-5);
        assert!((srgb_oetf(0.18) - 0.461356).abs() < 1e-5);
        assert_eq!(srgb8_to_linear(0), 0.0);
        assert_eq!(linear_to_srgb8(1.0), 255);
        assert_eq!(linear_to_srgb8(2.0), 255);
        assert_eq!(linear_to_srgb8(-1.0), 0);
    }

    #[test]
    fn every_code_survives_the_round_trip() {
        for code in 0..=255u8 {
            assert_eq!(linear_to_srgb8(srgb8_to_linear(code)), code);
        }
    }
}
