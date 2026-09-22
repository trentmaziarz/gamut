//! Shift edge, Feather and Contrast on the session, walked the way the window
//! walks them: a drag of one of their sliders is one undo step, the file
//! carries them, and a version and a preset leave them where they belong.
//! The window is left for feel.

use std::time::Instant;

use gamut_app::app::{OpenPhoto, Session};
use gamut_app::sidecar::load;
use gamut_app::undo::Settle;
use gamut_core::mask::{Edge, RadialGradient, Refine};
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

/// A session with one refined radial mask that lifts the exposure.
fn with_a_mask() -> Session {
    let mut session = Session::default();
    let mut mask = Mask::new("Roofs", MaskSource::Radial(RadialGradient::default()));
    mask.adjust.exposure = 1.0;
    mask.refine.amount = 100.0;
    session.edit.masks.push(mask);
    session.begin_history();
    session
}

#[test]
fn one_drag_of_each_edge_slider_is_one_undo_step() {
    let mut session = with_a_mask();
    for frame in 1..=40 {
        session.edit.masks[0].edge.shift = frame as f32 * -0.00025;
        session.mark_edited();
        drag_frame(&mut session);
        assert_eq!(undo_steps(&session), 0, "no step while the pointer is down");
    }
    settle(&mut session);
    assert_eq!(undo_steps(&session), 1, "the Shift edge drag");
    for frame in 1..=40 {
        session.edit.masks[0].edge.feather = frame as f32 * 0.00025;
        session.mark_edited();
        drag_frame(&mut session);
        assert_eq!(undo_steps(&session), 1, "no step while the pointer is down");
    }
    settle(&mut session);
    assert_eq!(undo_steps(&session), 2, "the Feather drag");
    for frame in 1..=50 {
        session.edit.masks[0].edge.contrast = frame as f32;
        session.mark_edited();
        drag_frame(&mut session);
        assert_eq!(undo_steps(&session), 2, "no step while the pointer is down");
    }
    settle(&mut session);
    assert_eq!(undo_steps(&session), 3, "one step a slider");
    let dragged = Edge {
        shift: 40.0 * -0.00025,
        feather: 40.0 * 0.00025,
        contrast: 50.0,
    };
    assert_eq!(session.edit.masks[0].edge, dragged);

    assert!(session.undo());
    assert_eq!(
        session.edit.masks[0].edge,
        Edge {
            contrast: 0.0,
            ..dragged
        }
    );
    assert!(session.undo());
    assert_eq!(
        session.edit.masks[0].edge,
        Edge {
            shift: dragged.shift,
            ..Edge::default()
        }
    );
    assert!(session.undo());
    assert_eq!(session.edit.masks[0].edge, Edge::default());
    assert_eq!(
        session.edit.masks[0].refine.amount, 100.0,
        "Refine edges is no part of an edge step"
    );
    assert!(session.develop_dirty, "the picture is stale");
    assert!(!session.undo(), "and nothing before it");
    assert!(session.redo() && session.redo() && session.redo());
    assert_eq!(session.edit.masks[0].edge, dragged);
}

#[test]
fn the_saved_sidecar_carries_the_edge_controls_and_follows_an_undo() {
    let dir = std::env::temp_dir().join("gamut-edge-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let photo = dir.join("edge.jpg");
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
    let plain = text(&mut session);
    assert!(plain.contains("\"refine\""), "{plain}");
    assert!(
        !plain.contains("\"edge\""),
        "the key is left out of the file while the three are at rest"
    );

    let edge = Edge {
        shift: -0.01,
        feather: 0.01,
        contrast: 50.0,
    };
    session.edit.masks[0].edge = edge;
    session.mark_edited();
    settle(&mut session);
    let saved = text(&mut session);
    assert!(saved.contains("\"version\": 3"), "{saved}");
    let value: serde_json::Value = serde_json::from_str(&saved).expect("json");
    let written = &value["edit"]["masks"][0]["edge"];
    let key = |name: &str| written[name].as_f64().map(|v| v as f32);
    assert_eq!(key("shift"), Some(-0.01), "{saved}");
    assert_eq!(key("feather"), Some(0.01), "{saved}");
    assert_eq!(key("contrast"), Some(50.0), "{saved}");
    assert!(
        !saved.contains("history") && !saved.contains("undo"),
        "{saved}"
    );
    let loaded = load(&photo).expect("the sidecar loads");
    assert_eq!(loaded, session.sidecar());
    assert_eq!(loaded.edit.masks[0].edge, edge);
    // A reload starts from that state, equal in everything.
    let mut fresh = Session::default();
    fresh.take_sidecar(loaded);
    assert_eq!(fresh.edit.masks[0].edge, edge);
    assert_eq!(fresh.edit.masks[0].refine, session.edit.masks[0].refine);

    assert!(session.undo());
    assert!(session.changed_at.is_some(), "the file is stale");
    let undone = text(&mut session);
    assert!(!undone.contains("\"edge\""), "undone on disk");
    assert_eq!(undone, plain, "the file is the one before the step");
    assert_eq!(load(&photo).expect("loads"), session.sidecar());
    assert!(session.redo());
    assert_eq!(text(&mut session), saved, "redone on disk");
}

#[test]
fn a_version_switch_and_a_preset_carry_the_edge_controls() {
    let mut session = with_a_mask();
    let edged = Edge {
        shift: 0.02,
        feather: 0.015,
        contrast: 70.0,
    };
    session.edit.masks[0].edge = edged;
    session
        .with_versions(|sidecar| sidecar.save_version("Edged"))
        .expect("save");
    session.edit.masks[0].edge = Edge::default();
    session
        .with_versions(|sidecar| {
            sidecar.active_version = None;
            sidecar.save_version("Plain")
        })
        .expect("save");
    settle(&mut session);
    session.request_switch("Edged").expect("a clean switch");
    assert_eq!(session.edit.masks[0].edge, edged, "the version has them");
    session.request_switch("Plain").expect("a clean switch");
    assert_eq!(session.edit.masks[0].edge, Edge::default());
    session.request_switch("Edged").expect("a clean switch");
    assert_eq!(session.edit.masks[0].edge, edged);
    assert_eq!(
        session.edit.masks[0].refine,
        Refine {
            amount: 100.0,
            ..Refine::default()
        },
        "and Refine edges beside them"
    );
    settle(&mut session);

    // A look preset holds adjustments: applied to the picture or to the
    // mask, it moves the look and carries the three through as they were.
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
    assert_eq!(session.edit.masks[0].edge, edged);
}
