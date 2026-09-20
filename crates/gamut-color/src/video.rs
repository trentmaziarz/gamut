//! The CPU twin of the video pass: a decoded YCbCr sample to linear
//! Rec.2020. The range is expanded, the BT.709 or BT.2020 matrix gives
//! R'G'B', the transfer is decoded (SDR through the sRGB curve the photos
//! use, HLG through the inverse OETF of BT.2100), and the primaries go to
//! Rec.2020. Every step mirrors `yuv_to_working.wgsl` line for line.
//!
//! The HLG path is provisional: the inverse OETF alone, normalised so that
//! a signal of 1.0 gives 1.0, clipped at 1.0, with no system gamma and no
//! tone map. PQ is decoded as HLG would be. T-8 replaces both with the
//! BT.2390 tone map.

use crate::matrices::{self, Mat3};
use crate::transfer::srgb_eotf;

/// The layout of a decoded frame's planes, which sets the bit depth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneFormat {
    /// 8 bit luma, then interleaved 8 bit Cb and Cr at half size.
    Nv12,
    /// 16 bit words holding 10 bit samples in their top bits, luma then
    /// interleaved Cb and Cr.
    P010,
}

impl PlaneFormat {
    /// Bytes per sample in each plane.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            PlaneFormat::Nv12 => 1,
            PlaneFormat::P010 => 2,
        }
    }

    /// The largest code the plane can hold: 255, or 1023 for the ten bits.
    pub fn max_code(self) -> u32 {
        match self {
            PlaneFormat::Nv12 => 255,
            PlaneFormat::P010 => 1023,
        }
    }

    /// The value a texture sampler returns for a code, which is how the
    /// shader sees it: the byte over 255, or the 16 bit word (the code
    /// shifted up six bits) over 65535.
    pub fn normalised(self, code: u32) -> f32 {
        match self {
            PlaneFormat::Nv12 => code as f32 / 255.0,
            PlaneFormat::P010 => (code << 6) as f32 / 65535.0,
        }
    }
}

/// The YCbCr matrix and the primaries of a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YuvSpace {
    /// BT.709, the SDR default; its primaries equal sRGB's.
    Bt709,
    /// BT.2020, what phone HDR clips carry.
    Bt2020,
}

/// The transfer function of a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    /// BT.709 or unspecified: decoded with the sRGB curve as the photos are.
    Sdr,
    /// Hybrid log gamma, ARIB STD-B67, what iPhones record.
    Hlg,
    /// PQ, SMPTE ST 2084. Decoded as HLG until T-8 lands the tone map.
    Pq,
}

impl Transfer {
    /// The id the shader switches on.
    pub fn id(self) -> u32 {
        match self {
            Transfer::Sdr => 0,
            Transfer::Hlg | Transfer::Pq => 1,
        }
    }
}

/// The colour tags of a stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoColour {
    pub space: YuvSpace,
    pub transfer: Transfer,
    /// Full range (0 to 255) rather than the limited 16 to 235.
    pub full_range: bool,
}

/// The range constants of a plane format in the shader's normalised units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range {
    /// The code of black.
    pub black: f32,
    /// The distance from black to white in luma.
    pub luma: f32,
    /// The distance from the lowest to the highest chroma code.
    pub chroma: f32,
    /// The code of neutral chroma.
    pub mid: f32,
}

/// The range constants: 16 to 235 and 16 to 240 at 8 bits, 64 to 940 and
/// 64 to 960 at 10 bits, or the whole range when the stream is full range.
pub fn range(format: PlaneFormat, full_range: bool) -> Range {
    let bits = match format {
        PlaneFormat::Nv12 => 8,
        PlaneFormat::P010 => 10,
    };
    let scale = 1u32 << (bits - 8);
    let (black, white, chroma_top) = if full_range {
        (0, format.max_code(), format.max_code())
    } else {
        (16 * scale, 235 * scale, 240 * scale)
    };
    let mid = 128 * scale;
    Range {
        black: format.normalised(black),
        luma: format.normalised(white) - format.normalised(black),
        chroma: format.normalised(chroma_top) - format.normalised(black),
        mid: format.normalised(mid),
    }
}

/// The matrix from expanded Y', Cb, Cr (Cb and Cr from -0.5 to 0.5) to
/// R'G'B', from the luma coefficients of the space.
pub fn yuv_matrix(space: YuvSpace) -> Mat3 {
    let (kr, kb) = match space {
        YuvSpace::Bt709 => (0.2126, 0.0722),
        YuvSpace::Bt2020 => (0.2627, 0.0593),
    };
    let kg = 1.0 - kr - kb;
    Mat3([
        [1.0, 0.0, 2.0 * (1.0 - kr)],
        [
            1.0,
            -2.0 * kb * (1.0 - kb) / kg,
            -2.0 * kr * (1.0 - kr) / kg,
        ],
        [1.0, 2.0 * (1.0 - kb), 0.0],
    ])
}

/// The matrix from the space's linear primaries to linear Rec.2020.
pub fn primaries_matrix(space: YuvSpace) -> Mat3 {
    match space {
        YuvSpace::Bt709 => matrices::srgb_to_rec2020(),
        YuvSpace::Bt2020 => Mat3::IDENTITY,
    }
}

/// Normalised Y, Cb, Cr samples (as the sampler returns them) to R'G'B'
/// in 0 to 1, clipped.
pub fn yuv_to_rgb(
    y: f32,
    cb: f32,
    cr: f32,
    matrix: YuvSpace,
    full_range: bool,
    format: PlaneFormat,
) -> [f32; 3] {
    let range = range(format, full_range);
    let yp = (y - range.black) / range.luma;
    let cb = (cb - range.mid) / range.chroma;
    let cr = (cr - range.mid) / range.chroma;
    yuv_matrix(matrix)
        .apply([yp, cb, cr])
        .map(|c| c.clamp(0.0, 1.0))
}

const HLG_A: f32 = 0.178_832_77;
const HLG_B: f32 = 0.284_668_92;
const HLG_C: f32 = 0.559_910_7;

/// The inverse OETF of BT.2100 hybrid log gamma: a signal in 0 to 1 to
/// scene light normalised so that 1.0 in gives 1.0 out (0.5 gives 1/12).
/// Clipped at 1.0.
pub fn hlg_inverse_oetf(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    let light = if v <= 0.5 {
        v * v / 3.0
    } else {
        (((v - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    };
    light.min(1.0)
}

/// One channel through the transfer function of the stream.
pub fn decode_transfer(v: f32, transfer: Transfer) -> f32 {
    match transfer {
        Transfer::Sdr => srgb_eotf(v),
        Transfer::Hlg | Transfer::Pq => hlg_inverse_oetf(v),
    }
}

/// Normalised Y, Cb, Cr to linear Rec.2020 through the ruled chain.
pub fn decode_video_pixel(px_yuv: [f32; 3], format: PlaneFormat, colour: VideoColour) -> [f32; 3] {
    let rgb = yuv_to_rgb(
        px_yuv[0],
        px_yuv[1],
        px_yuv[2],
        colour.space,
        colour.full_range,
        format,
    );
    let linear = rgb.map(|c| decode_transfer(c, colour.transfer));
    primaries_matrix(colour.space).apply(linear)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SDR: VideoColour = VideoColour {
        space: YuvSpace::Bt709,
        transfer: Transfer::Sdr,
        full_range: false,
    };

    #[test]
    fn black_and_white_codes_map_to_zero_and_one() {
        let f = PlaneFormat::Nv12;
        let black = decode_video_pixel(
            [f.normalised(16), f.normalised(128), f.normalised(128)],
            f,
            SDR,
        );
        for c in black {
            assert!(c.abs() < 1e-5, "{black:?}");
        }
        let white = decode_video_pixel(
            [f.normalised(235), f.normalised(128), f.normalised(128)],
            f,
            SDR,
        );
        for c in white {
            assert!((c - 1.0).abs() < 1e-4, "{white:?}");
        }
        let f = PlaneFormat::P010;
        let white = decode_video_pixel(
            [f.normalised(940), f.normalised(512), f.normalised(512)],
            f,
            SDR,
        );
        for c in white {
            assert!((c - 1.0).abs() < 1e-4, "{white:?}");
        }
        let full = VideoColour {
            full_range: true,
            ..SDR
        };
        let white = decode_video_pixel(
            [f.normalised(1023), f.normalised(512), f.normalised(512)],
            f,
            full,
        );
        for c in white {
            assert!((c - 1.0).abs() < 1e-4, "{white:?}");
        }
        let over = yuv_to_rgb(1.0, 0.5, 0.5, YuvSpace::Bt709, false, PlaneFormat::Nv12);
        assert_eq!(over, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_bt709_grey_stays_neutral() {
        let f = PlaneFormat::Nv12;
        let grey = decode_video_pixel(
            [f.normalised(126), f.normalised(128), f.normalised(128)],
            f,
            SDR,
        );
        assert!(
            (grey[0] - grey[1]).abs() < 1e-5 && (grey[1] - grey[2]).abs() < 1e-5,
            "{grey:?}"
        );
        let expected = srgb_eotf((126.0 / 255.0 - 16.0 / 255.0) / (219.0 / 255.0));
        assert!((grey[1] - expected).abs() < 1e-4, "{grey:?} vs {expected}");
    }

    #[test]
    fn hlg_half_signal_is_one_twelfth() {
        assert!((hlg_inverse_oetf(0.5) - 1.0 / 12.0).abs() < 1e-6);
        assert!((hlg_inverse_oetf(1.0) - 1.0).abs() < 1e-5);
        assert_eq!(hlg_inverse_oetf(0.0), 0.0);
        assert_eq!(hlg_inverse_oetf(1.5), 1.0);
        let below = hlg_inverse_oetf(0.4999);
        let above = hlg_inverse_oetf(0.5001);
        assert!(above > below && above - below < 1e-3);
    }

    #[test]
    fn the_matrices_turn_the_primaries_white_and_red_back() {
        for space in [YuvSpace::Bt709, YuvSpace::Bt2020] {
            let white = yuv_matrix(space).apply([1.0, 0.0, 0.0]);
            assert_eq!(white, [1.0, 1.0, 1.0]);
        }
        // BT.709 red: Y' 0.2126, Cb -0.2126/1.8556, Cr 0.5.
        let red = yuv_matrix(YuvSpace::Bt709).apply([0.2126, -0.2126 / 1.8556, 0.5]);
        assert!(
            (red[0] - 1.0).abs() < 1e-4 && red[1].abs() < 1e-4 && red[2].abs() < 1e-4,
            "{red:?}"
        );
        let ten = range(PlaneFormat::P010, false);
        assert!((ten.black - 4096.0 / 65535.0).abs() < 1e-7);
        assert!((ten.luma - (60160.0 - 4096.0) / 65535.0).abs() < 1e-6);
    }
}
