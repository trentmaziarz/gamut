//! Bradford chromatic adaptation: the matrix that moves colours seen under
//! one white so that they appear as they would under another.

use crate::matrices::Mat3;

/// XYZ to the Bradford cone response.
pub const BRADFORD: Mat3 = Mat3([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
]);

/// The XYZ to XYZ matrix that adapts from `source_white` to
/// `destination_white`, both as XYZ with Y = 1.
pub fn adaptation(source_white: [f64; 3], destination_white: [f64; 3]) -> Mat3 {
    let s = BRADFORD.apply_f64(source_white);
    let d = BRADFORD.apply_f64(destination_white);
    let scale = Mat3::diagonal([d[0] / s[0], d[1] / s[1], d[2] / s[2]]);
    BRADFORD.inverse() * scale * BRADFORD
}

#[cfg(test)]
mod tests {
    use super::adaptation;
    use crate::matrices::{Mat3, xyz_from_xy};

    const D65: (f64, f64) = (0.31272, 0.32903);
    const D50: (f64, f64) = (0.34567, 0.35850);

    #[test]
    fn d65_to_d65_is_the_identity() {
        let w = xyz_from_xy(D65.0, D65.1);
        let m = adaptation(w, w);
        for r in 0..3 {
            for c in 0..3 {
                assert!((m.0[r][c] - Mat3::IDENTITY.0[r][c]).abs() < 1e-6, "{m:?}");
            }
        }
    }

    #[test]
    fn the_source_white_lands_on_the_destination_white() {
        let d65 = xyz_from_xy(D65.0, D65.1);
        let d50 = xyz_from_xy(D50.0, D50.1);
        let adapted = adaptation(d50, d65).apply_f64(d50);
        for (a, b) in adapted.iter().zip(d65) {
            assert!((a - b).abs() < 1e-9, "{adapted:?} is not {d65:?}");
        }
    }

    #[test]
    fn d50_to_d65_matches_the_published_matrix() {
        let m = adaptation(xyz_from_xy(D50.0, D50.1), xyz_from_xy(D65.0, D65.1));
        let published = [
            [0.9555766, -0.0230393, 0.0631636],
            [-0.0282895, 1.0099416, 0.0210077],
            [0.0122982, -0.0204830, 1.3299098],
        ];
        for (row, expected) in m.0.iter().zip(published) {
            for (value, wanted) in row.iter().zip(expected) {
                assert!((value - wanted).abs() < 1e-3, "{m:?}");
            }
        }
    }
}
