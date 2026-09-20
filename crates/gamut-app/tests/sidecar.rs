//! Writes and reads a sidecar for a copy of a fixture, as the app does.

use gamut_app::sidecar::{load, save};
use gamut_core::{Adjustments, Crop, CropAspect, CropRect, PhotoEdit, Sidecar};
use gamut_media::fixtures;

#[test]
fn a_sidecar_round_trips_next_to_a_copied_fixture() {
    let dir = std::env::temp_dir().join("gamut-sidecar-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let photo = dir.join("Portrait_8.jpg");
    std::fs::copy(fixtures::require("Portrait_8.jpg"), &photo).expect("copy the fixture");
    let _ = std::fs::remove_file(Sidecar::path_for(&photo));
    assert_eq!(load(&photo), None, "no sidecar before the first save");

    let sidecar = Sidecar::new(
        PhotoEdit::from(Adjustments {
            white_balance_temperature: 12.0,
            exposure: -0.4,
            vibrance: 25.0,
            ..Adjustments::default()
        }),
        Crop {
            aspect: CropAspect::Story9x16,
            rect: CropRect::fitted(CropAspect::Story9x16, 1200, 1800),
        },
    );
    let path = save(&photo, &sidecar).expect("save");
    assert_eq!(path.file_name().unwrap(), "Portrait_8.jpg.slate.json");
    assert!(path.is_file());
    assert_eq!(load(&photo), Some(sidecar));
}
