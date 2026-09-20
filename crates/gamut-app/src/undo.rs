//! Undo and redo of the open file. The history itself is
//! [`gamut_core::History`]; this module says what a state of the open file
//! is, when a change has settled into a step, and which keys ask for a step
//! back or forward. The session takes a state back in
//! [`Session::undo`](crate::app::Session::undo).
//!
//! What the history holds is what the file saves: a photo's whole sidecar
//! (the edit with its masks, the crop, the versions and the active one), or
//! a project's edit, crop and track. The view, the selected mask, the overlay
//! and the playhead are not part of it.

use std::time::Duration;

use gamut_core::{Crop, PhotoEdit, Sidecar, Track};

/// How long after a change from the arrow keys a step is taken, so a run of
/// key presses on a slider is one step.
pub const KEY_SETTLE: Duration = Duration::from_millis(400);

/// A state of the open file.
#[derive(Clone, Debug, PartialEq)]
pub enum Snapshot {
    Photo(Box<Sidecar>),
    Project {
        edit: Box<PhotoEdit>,
        crop: Crop,
        track: Track,
    },
}

/// What the window knows this frame about a change being under way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Settle {
    /// The pointer is down on a widget: a drag is under way.
    pub pointer_busy: bool,
    /// A text field or a drag value has the keyboard.
    pub typing: bool,
    /// An arrow key is down: a slider may be stepping.
    pub arrow_key: bool,
}

/// Whether a change has settled into a step: not while the pointer is down
/// on a widget, not while someone types, and not until [`KEY_SETTLE`] after
/// the last change that came from the arrow keys.
pub fn settled(input: Settle, since_key_change: Option<Duration>) -> bool {
    !input.pointer_busy
        && !input.typing
        && since_key_change.is_none_or(|waited| waited >= KEY_SETTLE)
}

/// A step the keys ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistoryKey {
    Undo,
    Redo,
}

/// The keys of the history as they are this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HistoryKeys {
    /// Ctrl, or Cmd on a Mac.
    pub command: bool,
    pub shift: bool,
    pub z: bool,
    pub y: bool,
    /// A text field has the keyboard: its own undo owns the keys.
    pub typing: bool,
    /// The Save, Discard, Cancel prompt is up.
    pub prompt: bool,
}

/// Ctrl+Z undoes; Ctrl+Shift+Z and Ctrl+Y redo. Neither acts while someone
/// types or while the prompt is up.
pub fn history_key(keys: HistoryKeys) -> Option<HistoryKey> {
    if !keys.command || keys.typing || keys.prompt {
        return None;
    }
    if keys.z {
        return Some(if keys.shift {
            HistoryKey::Redo
        } else {
            HistoryKey::Undo
        });
    }
    (keys.y && !keys.shift).then_some(HistoryKey::Redo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_change_settles_when_the_pointer_the_text_and_the_keys_are_done() {
        assert!(settled(Settle::default(), None));
        let dragging = Settle {
            pointer_busy: true,
            ..Settle::default()
        };
        assert!(!settled(dragging, None));
        let typing = Settle {
            typing: true,
            ..Settle::default()
        };
        assert!(!settled(typing, None));
        // After an arrow key moved a slider the step waits 400 ms.
        assert!(!settled(Settle::default(), Some(Duration::from_millis(0))));
        assert!(!settled(
            Settle::default(),
            Some(Duration::from_millis(399))
        ));
        assert!(settled(Settle::default(), Some(KEY_SETTLE)));
        assert!(!settled(dragging, Some(Duration::from_secs(2))));
    }

    #[test]
    fn ctrl_z_undoes_and_ctrl_shift_z_and_ctrl_y_redo() {
        let ctrl = HistoryKeys {
            command: true,
            ..HistoryKeys::default()
        };
        assert_eq!(
            history_key(HistoryKeys { z: true, ..ctrl }),
            Some(HistoryKey::Undo)
        );
        assert_eq!(
            history_key(HistoryKeys {
                z: true,
                shift: true,
                ..ctrl
            }),
            Some(HistoryKey::Redo)
        );
        assert_eq!(
            history_key(HistoryKeys { y: true, ..ctrl }),
            Some(HistoryKey::Redo)
        );
        assert_eq!(history_key(ctrl), None);
        // A bare Z or Y is a letter.
        assert_eq!(
            history_key(HistoryKeys {
                z: true,
                ..HistoryKeys::default()
            }),
            None
        );
        assert_eq!(
            history_key(HistoryKeys {
                y: true,
                ..HistoryKeys::default()
            }),
            None
        );
    }

    #[test]
    fn the_history_keys_do_nothing_while_typing_or_under_the_prompt() {
        let undo = HistoryKeys {
            command: true,
            z: true,
            ..HistoryKeys::default()
        };
        assert_eq!(
            history_key(HistoryKeys {
                typing: true,
                ..undo
            }),
            None,
            "the text field's own undo owns the keys"
        );
        assert_eq!(
            history_key(HistoryKeys {
                prompt: true,
                ..undo
            }),
            None
        );
    }
}
