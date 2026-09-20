//! Saves two named versions next to a copy of a fixture, reads the sidecar
//! back and renders each version by its name, as `--version-name` does. The
//! renders skip, and say so, when the machine has no adapter.

use slate_app::headless::{EditSource, HeadlessError};
use slate_app::screenshot::write_developed;
use slate_app::sidecar::{load, save};
use slate_core::Sidecar;
use slate_core::look::Wheel;
use slate_media::fixtures;

#[test]
fn two_versions_survive_a_reload_and_render_by_name() {
    let dir = std::env::temp_dir().join("slate-versions-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let photo = dir.join("Portrait_8.jpg");
    std::fs::copy(fixtures::require("Portrait_8.jpg"), &photo).expect("copy the fixture");
    let _ = std::fs::remove_file(Sidecar::path_for(&photo));

    let mut sidecar = Sidecar::default();
    sidecar.edit.exposure = 0.8;
    sidecar.edit.look.wheels.shadows = Wheel {
        x: -0.3,
        y: -0.2,
        luminance: 0.0,
    };
    sidecar.save_version("Teal").expect("the first version");
    sidecar.edit = Default::default();
    sidecar.edit.exposure = -0.8;
    sidecar.edit.clarity = 40.0;
    sidecar.active_version = None;
    sidecar.save_version("Dark").expect("the second version");
    sidecar.edit = Default::default();

    let path = save(&photo, &sidecar).expect("save");
    let text = std::fs::read_to_string(&path).expect("read");
    assert!(text.contains("\"version\": 2"), "{text}");
    let back = load(&photo).expect("the sidecar loads");
    assert_eq!(back, sidecar);
    assert_eq!(back.versions.len(), 2);
    assert_eq!(back.version("teal").expect("found").edit.exposure, 0.8);

    let render = |version: Option<&str>, name: &str| -> Result<Vec<u8>, HeadlessError> {
        let out = dir.join(name);
        let source = EditSource {
            version,
            ..EditSource::default()
        };
        write_developed(&photo, source, &out)?;
        Ok(std::fs::read(&out).expect("the PNG was written"))
    };
    let teal = match render(Some("Teal"), "teal.png") {
        Err(HeadlessError::NoAdapter) => {
            println!("no adapter, skipped");
            return;
        }
        other => other.expect("the Teal version renders"),
    };
    let dark = render(Some("dark"), "dark.png").expect("the Dark version renders");
    let working = render(None, "working.png").expect("the working state renders");
    assert_ne!(teal, dark);
    assert_ne!(teal, working);
    assert_ne!(dark, working);

    let error = render(Some("Nope"), "nope.png").expect_err("an unknown name is refused");
    assert_eq!(
        error.to_string(),
        "no version named Nope: the versions are Teal, Dark"
    );
}
