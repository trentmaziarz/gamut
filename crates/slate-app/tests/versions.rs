//! Saves two named versions next to a copy of a fixture, reads the sidecar
//! back and renders each version by its name, as `--version-name` does. The
//! renders skip, and say so, when the machine has no adapter.
//!
//! The other tests walk the session through what a person does at the
//! Versions section: save, change, switch away and back, answer the prompt,
//! update, close and reload.

use slate_app::app::{Session, SwitchAnswer};
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
    assert!(text.contains("\"version\": 3"), "{text}");
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

/// A session with two versions, Sunset (exposure 1) and Normal (exposure 0),
/// working from Normal.
fn sunset_and_normal() -> Session {
    let mut session = Session::default();
    session.edit.exposure = 1.0;
    session
        .with_versions(|sidecar| sidecar.save_version("Sunset"))
        .expect("save");
    session.edit = Default::default();
    session
        .with_versions(|sidecar| sidecar.save_version("Normal"))
        .expect("save");
    session
}

/// The session a reload of the sidecar gives.
fn reloaded(session: &Session) -> Session {
    let sidecar = Sidecar::from_json(&session.sidecar().to_json()).expect("parse");
    let mut fresh = Session::default();
    fresh.take_sidecar(sidecar);
    fresh
}

#[test]
fn a_switch_on_a_clean_state_needs_no_prompt() {
    let mut session = sunset_and_normal();
    assert!(!session.version_is_dirty());
    session.request_switch("Sunset").expect("switch");
    assert_eq!(session.adjust.pending_switch, None);
    assert_eq!(session.edit.exposure, 1.0);
    assert_eq!(session.active_version.as_deref(), Some("Sunset"));
    assert_eq!(session.status.as_deref(), Some("Switched to Sunset."));
    assert!(session.develop_dirty && session.changed_at.is_some());
}

#[test]
fn a_switch_over_unsaved_work_waits_for_an_answer() {
    let mut session = sunset_and_normal();
    session.edit.contrast = 30.0;
    assert!(session.version_is_dirty());
    session
        .request_switch("Sunset")
        .expect("the prompt goes up");
    assert_eq!(session.adjust.pending_switch.as_deref(), Some("Sunset"));
    assert_eq!(session.edit.contrast, 30.0, "nothing moved yet");
    assert_eq!(session.active_version.as_deref(), Some("Normal"));

    // Cancel stays on the working state and keeps the work.
    session.answer_switch(SwitchAnswer::Cancel).expect("cancel");
    assert_eq!(session.adjust.pending_switch, None);
    assert_eq!(session.edit.contrast, 30.0);
    assert_eq!(session.active_version.as_deref(), Some("Normal"));
    assert_eq!(session.versions[1].edit.contrast, 0.0);
}

#[test]
fn discard_switches_and_leaves_every_version_as_saved() {
    let mut session = sunset_and_normal();
    let saved = session.versions.clone();
    session.edit.contrast = 30.0;
    session.request_switch("Sunset").expect("prompt");
    session
        .answer_switch(SwitchAnswer::Discard)
        .expect("discard");
    assert_eq!(session.edit.exposure, 1.0);
    assert_eq!(session.edit.contrast, 0.0);
    assert_eq!(session.versions, saved);
    assert_eq!(
        session.status.as_deref(),
        Some("Discarded the changes, switched to Sunset.")
    );

    // Back on Normal, the work is gone: it was never saved.
    session.request_switch("Normal").expect("clean switch");
    assert_eq!(session.edit.contrast, 0.0);
    assert_eq!(reloaded(&session).versions, saved);
}

#[test]
fn save_keeps_the_work_in_the_active_version_then_switches() {
    let mut session = sunset_and_normal();
    session.edit.contrast = 30.0;
    session.request_switch("Sunset").expect("prompt");
    session.answer_switch(SwitchAnswer::Save).expect("save");
    assert_eq!(session.active_version.as_deref(), Some("Sunset"));
    assert_eq!(session.edit.exposure, 1.0);
    assert_eq!(
        session.versions[1].edit.contrast, 30.0,
        "Normal took the work"
    );
    assert_eq!(session.versions[0].edit.contrast, 0.0, "Sunset did not");
    assert_eq!(
        session.status.as_deref(),
        Some("Saved into Normal, switched to Sunset.")
    );

    session.request_switch("Normal").expect("clean switch");
    assert_eq!(session.edit.contrast, 30.0);
}

#[test]
fn update_writes_a_version_and_survives_a_reload() {
    let mut session = sunset_and_normal();
    session.edit.clarity = 25.0;
    session.crop.rect.width = 0.5;
    session.update_version("Normal").expect("update");
    assert!(!session.version_is_dirty());
    assert_eq!(session.status.as_deref(), Some("Updated Normal."));

    // Close and reload: the versions, the active one and the work are there.
    let fresh = reloaded(&session);
    assert_eq!(fresh.versions, session.versions);
    assert_eq!(fresh.active_version.as_deref(), Some("Normal"));
    assert_eq!(fresh.versions[1].edit.clarity, 25.0);
    assert_eq!(fresh.versions[1].crop.rect.width, 0.5);
    assert_eq!(fresh.versions[0].edit.clarity, 0.0);
    assert!(!fresh.version_is_dirty());

    // Unsaved work survives a reload as unsaved work.
    session.edit.clarity = 60.0;
    let fresh = reloaded(&session);
    assert!(fresh.version_is_dirty());
    assert_eq!(fresh.edit.clarity, 60.0);
    assert_eq!(fresh.versions[1].edit.clarity, 25.0);
}

#[test]
fn a_switch_to_an_unknown_version_raises_no_prompt() {
    let mut session = sunset_and_normal();
    session.edit.contrast = 30.0;
    let error = session.request_switch("Nope").expect_err("unknown");
    assert_eq!(
        error.to_string(),
        "no version named Nope: the versions are Sunset, Normal"
    );
    assert_eq!(session.adjust.pending_switch, None);
    assert_eq!(session.edit.contrast, 30.0);
}
