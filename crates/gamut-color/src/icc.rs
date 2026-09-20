//! ICC profiles through moxcms: which input transform a photo needs to reach
//! the working space. Phone photos carry either no profile (sRGB), the
//! Display P3 profile that iPhones write, or some other RGB matrix profile.

use moxcms::ColorProfile;

/// The colour space a decoded photo is in. The transfer function is taken
/// to be the sRGB curve in every case; only the primaries differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SourceSpace {
    /// sRGB primaries, D65. The default when a file carries no profile.
    Srgb,
    /// Display P3 primaries, D65. What iPhones write in JPEG and HEIC.
    DisplayP3,
    /// Any other RGB matrix profile. The matrix takes the source's linear
    /// RGB to linear Rec.2020, row major, and comes from the profile.
    Icc { to_rec2020: [[f32; 3]; 3] },
}

/// How close each XYZ coordinate of a colorant has to be to count as the
/// same primaries.
const PRIMARY_TOLERANCE: f64 = 0.001;

/// Reads an ICC profile and decides which source space it describes. A
/// profile that does not parse is treated as sRGB, with a log line.
pub fn source_space(icc: &[u8]) -> SourceSpace {
    let profile = match ColorProfile::new_from_slice(icc) {
        Ok(profile) => profile,
        Err(error) => {
            log::warn!("ICC profile did not parse ({error:?}); treating the photo as sRGB");
            return SourceSpace::Srgb;
        }
    };
    if primaries_match(&profile, &ColorProfile::new_srgb()) {
        log::info!("ICC profile has sRGB primaries");
        return SourceSpace::Srgb;
    }
    if primaries_match(&profile, &ColorProfile::new_display_p3()) {
        log::info!("ICC profile has Display P3 primaries");
        return SourceSpace::DisplayP3;
    }
    let matrix = profile.transform_matrix(&ColorProfile::new_bt2020());
    let to_rec2020 = matrix.v.map(|row| row.map(|value| value as f32));
    log::info!("ICC profile is neither sRGB nor Display P3; using its own matrix to Rec.2020");
    SourceSpace::Icc { to_rec2020 }
}

fn primaries_match(a: &ColorProfile, b: &ColorProfile) -> bool {
    let pairs = [
        (a.red_colorant, b.red_colorant),
        (a.green_colorant, b.green_colorant),
        (a.blue_colorant, b.blue_colorant),
    ];
    pairs.iter().all(|(p, q)| {
        (p.x - q.x).abs() <= PRIMARY_TOLERANCE
            && (p.y - q.y).abs() <= PRIMARY_TOLERANCE
            && (p.z - q.z).abs() <= PRIMARY_TOLERANCE
    })
}

#[cfg(test)]
mod tests {
    use super::{SourceSpace, source_space};
    use moxcms::ColorProfile;

    #[test]
    fn an_srgb_profile_is_srgb() {
        let bytes = ColorProfile::new_srgb().encode().expect("encode sRGB");
        assert_eq!(source_space(&bytes), SourceSpace::Srgb);
    }

    #[test]
    fn a_display_p3_profile_is_display_p3() {
        let bytes = ColorProfile::new_display_p3()
            .encode()
            .expect("encode Display P3");
        assert_eq!(source_space(&bytes), SourceSpace::DisplayP3);
    }

    #[test]
    fn a_rec2020_profile_gets_a_matrix_near_identity() {
        let bytes = ColorProfile::new_bt2020()
            .encode()
            .expect("encode Rec.2020");
        let SourceSpace::Icc { to_rec2020 } = source_space(&bytes) else {
            panic!("Rec.2020 is not sRGB or Display P3");
        };
        for (r, row) in to_rec2020.iter().enumerate() {
            for (c, value) in row.iter().enumerate() {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-3,
                    "matrix {to_rec2020:?} is not the identity"
                );
            }
        }
    }

    #[test]
    fn garbage_falls_back_to_srgb() {
        assert_eq!(source_space(b"not a profile"), SourceSpace::Srgb);
    }
}
