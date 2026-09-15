//! 3 by 3 matrices and the RGB to XYZ matrices of the spaces M1 meets. All
//! three are the D65 versions, so no adaptation sits between them.

use crate::SourceSpace;

/// A 3 by 3 matrix, row major, kept in f64 so that products and inverses
/// hold to well under 1e-6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    /// A diagonal matrix.
    pub const fn diagonal(d: [f64; 3]) -> Mat3 {
        Mat3([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]])
    }

    /// From f32 rows, as an ICC matrix arrives.
    pub fn from_rows_f32(rows: [[f32; 3]; 3]) -> Mat3 {
        Mat3(rows.map(|row| row.map(f64::from)))
    }

    /// `self` times `other`: apply `other` first, then `self`.
    pub fn mul(self, other: Mat3) -> Mat3 {
        let mut out = [[0.0; 3]; 3];
        for (r, row) in out.iter_mut().enumerate() {
            for (c, value) in row.iter_mut().enumerate() {
                *value = (0..3).map(|k| self.0[r][k] * other.0[k][c]).sum();
            }
        }
        Mat3(out)
    }

    /// The matrix applied to a column vector.
    pub fn apply_f64(self, v: [f64; 3]) -> [f64; 3] {
        self.0
            .map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
    }

    /// The matrix applied to an f32 pixel, computed in f64.
    pub fn apply(self, v: [f32; 3]) -> [f32; 3] {
        self.apply_f64(v.map(f64::from)).map(|x| x as f32)
    }

    pub fn transpose(self) -> Mat3 {
        let m = self.0;
        Mat3([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    /// The inverse by cofactors. Panics on a singular matrix, which none of
    /// the colour matrices are.
    pub fn inverse(self) -> Mat3 {
        let m = self.0;
        let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
        let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
        let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
        let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
        assert!(det.abs() > 1e-18, "singular matrix {m:?}");
        let inv = 1.0 / det;
        Mat3([
            [
                c00 * inv,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
            ],
            [
                c01 * inv,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
            ],
            [
                c02 * inv,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
            ],
        ])
    }

    /// The rows as f32.
    pub fn to_rows_f32(self) -> [[f32; 3]; 3] {
        self.0.map(|row| row.map(|x| x as f32))
    }

    /// The layout of a `mat3x3<f32>` in a WGSL uniform block: three
    /// columns, each a vec3 padded to 16 bytes.
    pub fn to_wgsl_columns(self) -> [[f32; 4]; 3] {
        let m = self.0;
        [0, 1, 2].map(|c| [m[0][c] as f32, m[1][c] as f32, m[2][c] as f32, 0.0])
    }
}

/// sRGB (and Rec.709) linear RGB to XYZ, D65.
pub const SRGB_TO_XYZ: Mat3 = Mat3([
    [0.4123908, 0.3575843, 0.1804808],
    [0.2126390, 0.7151687, 0.0721923],
    [0.0193308, 0.1191948, 0.9505322],
]);

/// Display P3 linear RGB to XYZ, D65.
pub const DISPLAY_P3_TO_XYZ: Mat3 = Mat3([
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
]);

/// Rec.2020 linear RGB to XYZ, D65.
pub const REC2020_TO_XYZ: Mat3 = Mat3([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
]);

pub fn xyz_to_srgb() -> Mat3 {
    SRGB_TO_XYZ.inverse()
}

pub fn xyz_to_display_p3() -> Mat3 {
    DISPLAY_P3_TO_XYZ.inverse()
}

pub fn xyz_to_rec2020() -> Mat3 {
    REC2020_TO_XYZ.inverse()
}

pub fn srgb_to_rec2020() -> Mat3 {
    xyz_to_rec2020().mul(SRGB_TO_XYZ)
}

pub fn display_p3_to_rec2020() -> Mat3 {
    xyz_to_rec2020().mul(DISPLAY_P3_TO_XYZ)
}

pub fn rec2020_to_srgb() -> Mat3 {
    xyz_to_srgb().mul(REC2020_TO_XYZ)
}

/// The matrix from a source space's linear RGB to linear Rec.2020.
pub fn input_matrix(space: SourceSpace) -> Mat3 {
    match space {
        SourceSpace::Srgb => srgb_to_rec2020(),
        SourceSpace::DisplayP3 => display_p3_to_rec2020(),
        SourceSpace::Icc { to_rec2020 } => Mat3::from_rows_f32(to_rec2020),
    }
}

/// XYZ with Y = 1 from an xy chromaticity.
pub fn xyz_from_xy(x: f64, y: f64) -> [f64; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_near(a: Mat3, b: Mat3, tolerance: f64) {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (a.0[r][c] - b.0[r][c]).abs() < tolerance,
                    "{a:?} differs from {b:?} at {r},{c}"
                );
            }
        }
    }

    #[test]
    fn a_matrix_times_its_inverse_is_the_identity() {
        for m in [SRGB_TO_XYZ, DISPLAY_P3_TO_XYZ, REC2020_TO_XYZ] {
            assert_near(m.mul(m.inverse()), Mat3::IDENTITY, 1e-9);
        }
    }

    #[test]
    fn srgb_round_trips_through_rec2020() {
        let there_and_back = rec2020_to_srgb().mul(srgb_to_rec2020());
        assert_near(there_and_back, Mat3::IDENTITY, 1e-6);
        let px = srgb_to_rec2020().apply([0.2, 0.5, 0.9]);
        let back = rec2020_to_srgb().apply(px);
        for (a, b) in back.iter().zip([0.2, 0.5, 0.9]) {
            assert!((a - b).abs() < 1e-6, "{back:?}");
        }
    }

    #[test]
    fn white_maps_to_white() {
        for m in [
            srgb_to_rec2020(),
            display_p3_to_rec2020(),
            rec2020_to_srgb(),
        ] {
            let w = m.apply([1.0, 1.0, 1.0]);
            for c in w {
                assert!((c - 1.0).abs() < 1e-4, "{w:?}");
            }
        }
    }

    #[test]
    fn wgsl_columns_are_the_transpose() {
        let m = Mat3([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(
            m.to_wgsl_columns(),
            [
                [1.0, 4.0, 7.0, 0.0],
                [2.0, 5.0, 8.0, 0.0],
                [3.0, 6.0, 9.0, 0.0]
            ]
        );
    }
}
