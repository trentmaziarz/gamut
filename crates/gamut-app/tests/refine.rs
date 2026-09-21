//! Refine edges on the session, walked the way the window walks it: a drag of
//! one of its sliders is one undo step, the file carries it, and a version
//! and a preset leave it where it belongs. The window is left for feel.

use std::time::Instant;

use gamut_app::app::{OpenPhoto, Session};
use gamut_app::sidecar::load;
use gamut_app::undo::Settle;
use gamut_core::mask::{RadialGradient, Refine};
use gamut_core::preset::Groups;
use gamut_core::{LookPreset, Mask, MaskSource, Sidecar};

/// A frame in which nothing is held: whatever changed is a step.
fn settle(session: &mut Session) {
    session.observe_history(Settle::default(), Instant::now());
}

/// A frame in the middle of a drag.
fn drag_frame(session: &mut Session) {
    let dragging = Settle {
        pointer_busy: true,
        ..Settle::default()
    };
    session.observe_history(dragging, Instant::now());
}

fn undo_steps(session: &Session) -> usize {
    session.history.as_ref().map_or(0, |h| h.undo_steps())
}

/// A session with one radial mask that lifts the exposure.
fn with_a_mask() -> Session {
    let mut session = Session::default();
    let mut mask = Mask::new("Sky", MaskSource::Radial(RadialGradient::default()));
    mask.adjust.exposure = 1.0;
    session.edit.masks.push(mask);
    session.begin_history();
    session
}

#[test]
fn one_drag_of_a_refine_slider_is_one_undo_step() {
    let mut session = with_a_mask();
    for frame in 1..=50 {
        session.edit.masks[0].refine.amount = frame as f32 * 2.0;
        session.mark_edited();
        drag_frame(&mut session);
        assert_eq!(undo_steps(&session), 0, "no step while the pointer is down");
    }
    settle(&mut session);
    assert_eq!(undo_steps(&session), 1, "the Refine edges drag");
    for frame in 1..=20 {
        session.edit.masks[0].refine.radius = 0.01 + frame as f32 * 0.001;
        session.mark_edited();
        drag_frame(&mut session);
    }
    settle(&mut session);
    for frame in 1..=20 {
        session.edit.masks[0].refine.sensitivity = 50.0 + frame as f32;
        session.mark_edited();
        drag_frame(&mut session);
    }
    settle(&mut session);
    assert_eq!(undo_steps(&session), 3, "one step a slider");
    let dragged = Refine {
        amount: 100.0,
        radius: 0.01 + 20.0 * 0.001,
        sensitivity: 70.0,
    };
    assert_eq!(session.edit.masks[0].refine, dragged);

    assert!(session.undo());
    assert_eq!(session.edit.masks[0].refine.sensitivity, 50.0);
    assert!(session.undo());
    assert_eq!(session.edit.masks[0].refine.radius, 0.01);
    assert!(session.undo());
    assert_eq!(session.edit.masks[0].refine, Refine::default());
    assert!(session.develop_dirty, "the picture is stale");
    assert!(!session.undo(), "and nothing before it");
    assert!(session.redo() && session.redo() && session.redo());
    assert_eq!(session.edit.masks[0].refine, dragged);
}

#[test]
fn the_saved_sidecar_carries_refine_edges_and_follows_an_undo() {
    let dir = std::env::temp_dir().join("gamut-refine-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let photo = dir.join("refine.jpg");
    let _ = std::fs::remove_file(Sidecar::path_for(&photo));
    let mut session = Session {
        photo: Some(OpenPhoto {
            path: photo.clone(),
            width: 6000,
            height: 4000,
        }),
        ..with_a_mask()
    };
    session.begin_history();
    let text = |session: &mut Session| -> String {
        session.save_now();
        std::fs::read_to_string(Sidecar::path_for(&photo)).expect("read")
    };
    session.mark_edited();
    assert!(
        !text(&mut session).contains("\"refine\""),
        "the key is left out of the file while Refine edges is off"
    );

    session.edit.masks[0].refine = Refine {
        amount: 80.0,
        radius: 0.025,
        sensitivity: 65.0,
    };
    session.mark_edited();
    settle(&mut session);
    let saved = text(&mut session);
    assert!(saved.contains("\"version\": 3"), "{saved}");
    assert!(saved.contains("\"refine\""), "{saved}");
    assert!(
        !saved.contains("history") && !saved.contains("undo"),
        "{saved}"
    );
    let loaded = load(&photo).expect("the sidecar loads");
    assert_eq!(loaded, session.sidecar());
    assert_eq!(loaded.edit.masks[0].refine.radius, 0.025);
    // A reload starts from that state, equal in everything.
    let mut fresh = Session::default();
    fresh.take_sidecar(loaded);
    assert_eq!(fresh.edit.masks[0].refine, session.edit.masks[0].refine);

    assert!(session.undo());
    assert!(session.changed_at.is_some(), "the file is stale");
    assert!(!text(&mut session).contains("\"refine\""), "undone on disk");
    assert_eq!(load(&photo).expect("loads"), session.sidecar());
    assert!(session.redo());
    assert!(text(&mut session).contains("\"refine\""), "redone on disk");
}

#[test]
fn a_version_carries_refine_edges_and_a_preset_leaves_it_alone() {
    let mut session = with_a_mask();
    let refined = Refine {
        amount: 100.0,
        radius: 0.04,
        sensitivity: 30.0,
    };
    session.edit.masks[0].refine = refined;
    session
        .with_versions(|sidecar| sidecar.save_version("Refined"))
        .expect("save");
    session.edit.masks[0].refine = Refine::default();
    session
        .with_versions(|sidecar| {
            sidecar.active_version = None;
            sidecar.save_version("Plain")
        })
        .expect("save");
    settle(&mut session);
    session.request_switch("Refined").expect("a clean switch");
    assert_eq!(session.edit.masks[0].refine, refined, "the version has it");
    session.request_switch("Plain").expect("a clean switch");
    assert_eq!(session.edit.masks[0].refine, Refine::default());
    session.request_switch("Refined").expect("a clean switch");
    settle(&mut session);

    // A look preset holds adjustments: applied to the picture or to the
    // mask, it moves the look and leaves Refine edges as it was.
    let look = gamut_core::Adjustments {
        exposure: 0.5,
        contrast: 20.0,
        ..Default::default()
    };
    let preset = LookPreset::from_edit(
        "Look",
        &look,
        Groups {
            basic: true,
            ..Groups::default()
        },
    );
    preset.apply(&mut session.edit.adjust);
    preset.apply(&mut session.edit.masks[0].adjust);
    session.mark_edited();
    settle(&mut session);
    assert_eq!(session.edit.adjust.contrast, 20.0);
    assert_eq!(session.edit.masks[0].adjust.contrast, 20.0);
    assert_eq!(session.edit.masks[0].refine, refined);
}
