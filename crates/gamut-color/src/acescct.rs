//! ACEScct, the log encoding the tone curves, the HSL mixer and the colour
//! wheels run in. The constants are those of the Academy specification
//! S-2016-001.
//!
//! The published encoding is defined on AP1 primaries. Gamut applies the
//! curve to linear Rec.2020 channels directly: the controls need the log
//! shape, and a gamut hop per pixel buys nothing visible.

/// Linear values at or below this take the linear segment.
pub const LINEAR_CUT: f32 = 0.007_812_5;

/// Encoded values at or below this decode through the linear segment.
pub const ENCODED_CUT: f32 = 0.155_251_14;

/// The slope of the linear segment.
pub const SLOPE: f32 = 10.540_237;

/// The offset of the linear segment, which is also the encoding of 0.
pub const OFFSET: f32 = 0.072_905_53;

/// The log segment is (log2(lin) + LOG_SHIFT) / LOG_SCALE.
pub const LOG_SHIFT: f32 = 9.72;

/// One stop is 1 / LOG_SCALE in the encoding.
pub const LOG_SCALE: f32 = 17.52;

/// The encoding of linear 0, the 0 of the normalised axis.
pub const BLACK: f32 = OFFSET;

/// The encoding of linear 1, the 1 of the normalised axis.
pub const WHITE: f32 = LOG_SHIFT / LOG_SCALE;

/// Linear to ACEScct. 0.18 encodes to 0.4136 and 1.0 to 0.5548.
pub fn encode(lin: f32) -> f32 {
    if lin <= LINEAR_CUT {
        SLOPE * lin + OFFSET
    } else {
        (lin.log2() + LOG_SHIFT) / LOG_SCALE
    }
}

/// ACEScct to linear, the inverse of [`encode`].
pub fn decode(v: f32) -> f32 {
    if v <= ENCODED_CUT {
        (v - OFFSET) / SLOPE
    } else {
        2f32.powf(v * LOG_SCALE - LOG_SHIFT)
    }
}

/// An encoded value on the axis of the tone curve: 0 at black, 1 at diffuse
/// white.
pub fn normalise(v: f32) -> f32 {
    (v - BLACK) / (WHITE - BLACK)
}

/// The inverse of [`normalise`].
pub fn denormalise(n: f32) -> f32 {
    n * (WHITE - BLACK) + BLACK
}

/// [`encode`] on every channel.
pub fn encode_pixel(px: [f32; 3]) -> [f32; 3] {
    px.map(encode)
}

/// [`decode`] on every channel.
pub fn decode_pixel(px: [f32; 3]) -> [f32; 3] {
    px.map(decode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_and_white_encode_to_the_published_values() {
        assert!((encode(0.18) - 0.4135).abs() < 1e-3, "{}", encode(0.18));
        assert!((encode(1.0) - 0.5548).abs() < 1e-3, "{}", encode(1.0));
        assert_eq!(encode(0.0), OFFSET);
    }

    #[test]
    fn the_two_segments_meet_at_the_cut() {
        let below = SLOPE * LINEAR_CUT + OFFSET;
        let above = (LINEAR_CUT.log2() + LOG_SHIFT) / LOG_SCALE;
        assert!((below - above).abs() < 1e-6, "{below} and {above}");
        assert!((below - ENCODED_CUT).abs() < 1e-6);
    }

    /// The round trip holds to 1e-6 in the encoding everywhere from 0 to 16.
    /// In linear light single precision cannot hold 1e-6 absolute at 16 (one
    /// ulp of the exponent is about 7e-7 relative), so the linear check is
    /// 1e-6 below 1 and 2e-6 relative above it.
    #[test]
    fn encode_and_decode_are_inverses_from_0_to_16() {
        for i in 0..=16_000 {
            let lin = i as f32 / 1000.0;
            let back = decode(encode(lin));
            let tolerance = 1e-6_f32.max(2e-6 * lin);
            assert!((back - lin).abs() <= tolerance, "{lin} came back as {back}");
            let v = encode(lin);
            assert!((encode(decode(v)) - v).abs() <= 1e-6, "{v} in the encoding");
        }
        let lin = -0.004;
        assert!((decode(encode(lin)) - lin).abs() <= 1e-6);
    }

    #[test]
    fn the_normalised_axis_runs_from_black_to_white() {
        assert_eq!(normalise(encode(0.0)), 0.0);
        assert!((normalise(encode(1.0)) - 1.0).abs() < 1e-6);
        for n in [-0.2, 0.0, 0.37, 1.0, 1.4] {
            assert!((normalise(denormalise(n)) - n).abs() < 1e-6);
        }
    }
}
