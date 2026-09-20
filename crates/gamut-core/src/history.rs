//! The undo history of an open file: states of the document, plain data with
//! no clock and no interface in it. The application shows the history every
//! state the document passes through and says whether the change has settled;
//! the history turns that into steps.
//!
//! A change that is still under way (a slider being dragged, a handle in the
//! hand) is not a step yet. It becomes one step when it settles, however many
//! states it passed through, so one drag is one undo. A change that ends where
//! it began is no step at all.
//!
//! The history lives as long as the file is open. It is never written
//! anywhere.

use std::collections::VecDeque;

/// The most steps a history keeps; the oldest is dropped past it.
pub const MAX_STEPS: usize = 500;

/// What [`History::observe`] saw.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Observed {
    /// The state differs from the one shown to the history last time.
    pub changed: bool,
    /// A step was taken: the state is the new present.
    pub stepped: bool,
}

/// The present state with the states before it and, after an undo, the
/// states after it.
#[derive(Clone, Debug)]
pub struct History<T: Clone + PartialEq> {
    present: T,
    /// The state last shown to [`observe`](Self::observe).
    seen: T,
    undo: VecDeque<T>,
    redo: Vec<T>,
    limit: usize,
}

impl<T: Clone + PartialEq> History<T> {
    /// An empty history of a file as it was opened.
    pub fn new(state: T) -> Self {
        Self::with_limit(state, MAX_STEPS)
    }

    /// The same with another cap on the steps, at least one.
    pub fn with_limit(state: T, limit: usize) -> Self {
        History {
            seen: state.clone(),
            present: state,
            undo: VecDeque::new(),
            redo: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Shows the history the state of the document. While `settled` is false
    /// a state that differs from the present is only pending. Once `settled`
    /// is true it becomes one step: the present goes onto the undo stack, the
    /// state becomes the present and the redo stack is cleared.
    pub fn observe(&mut self, state: &T, settled: bool) -> Observed {
        let changed = *state != self.seen;
        if changed {
            self.seen = state.clone();
        }
        let stepped = settled && *state != self.present;
        if stepped {
            self.step(state.clone());
        }
        Observed { changed, stepped }
    }

    fn step(&mut self, state: T) {
        let before = std::mem::replace(&mut self.present, state);
        self.undo.push_back(before);
        while self.undo.len() > self.limit {
            self.undo.pop_front();
        }
        self.redo.clear();
    }

    /// A pending change is a step before anything is taken back, so nothing
    /// a person did is lost to an undo that came before it settled.
    fn settle(&mut self, current: &T) {
        if *current != self.present {
            self.step(current.clone());
        }
        self.seen = current.clone();
    }

    /// The state to take for an undo, or `None` on an empty stack.
    /// `current` is the document as it is now.
    pub fn undo(&mut self, current: &T) -> Option<T> {
        self.settle(current);
        let before = self.undo.pop_back()?;
        let undone = std::mem::replace(&mut self.present, before);
        self.redo.push(undone);
        self.seen = self.present.clone();
        Some(self.present.clone())
    }

    /// The state to take for a redo, or `None` when nothing was undone, or
    /// when a new step has cleared what was.
    pub fn redo(&mut self, current: &T) -> Option<T> {
        self.settle(current);
        let after = self.redo.pop()?;
        let before = std::mem::replace(&mut self.present, after);
        self.undo.push_back(before);
        self.seen = self.present.clone();
        Some(self.present.clone())
    }

    /// Whether `state` differs from the one last shown to
    /// [`observe`](Self::observe): the document changed since then.
    pub fn changed(&self, state: &T) -> bool {
        *state != self.seen
    }

    /// The state the history holds as the present.
    pub fn present(&self) -> &T {
        &self.present
    }

    /// How many steps an undo can take back.
    pub fn undo_steps(&self) -> usize {
        self.undo.len()
    }

    /// How many steps a redo can bring back.
    pub fn redo_steps(&self) -> usize {
        self.redo.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_change_is_a_step() {
        let mut history = History::new(0);
        assert_eq!(
            history.observe(&1, true),
            Observed {
                changed: true,
                stepped: true
            }
        );
        assert_eq!((history.undo_steps(), history.redo_steps()), (1, 0));
        assert_eq!(history.present(), &1);
        assert_eq!(history.undo(&1), Some(0));
    }

    #[test]
    fn an_unsettled_run_of_changes_is_one_step_when_it_settles() {
        let mut history = History::new(0);
        for value in 1..=40 {
            let seen = history.observe(&value, false);
            assert!(seen.changed && !seen.stepped);
        }
        assert_eq!(history.undo_steps(), 0, "nothing until it settles");
        // The frame the pointer lets go: the same state, now settled.
        let seen = history.observe(&40, true);
        assert!(!seen.changed && seen.stepped);
        assert_eq!(history.undo_steps(), 1);
        assert_eq!(history.undo(&40), Some(0), "the whole drag in one undo");
        assert_eq!(history.redo(&0), Some(40));
    }

    #[test]
    fn a_change_equal_to_the_present_is_nothing() {
        let mut history = History::new(7);
        assert_eq!(history.observe(&7, true), Observed::default());
        // A drag that ends where it began.
        history.observe(&9, false);
        history.observe(&7, false);
        assert!(!history.observe(&7, true).stepped);
        assert_eq!(history.undo_steps(), 0);
        assert_eq!(history.undo(&7), None);
    }

    #[test]
    fn undo_and_redo_walk_both_ways() {
        let mut history = History::new(0);
        for value in 1..=3 {
            history.observe(&value, true);
        }
        assert_eq!(history.undo(&3), Some(2));
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!((history.undo_steps(), history.redo_steps()), (1, 2));
        assert_eq!(history.redo(&1), Some(2));
        assert_eq!(history.redo(&2), Some(3));
        assert_eq!(history.redo(&3), None);
        assert_eq!(history.undo(&3), Some(2));
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!(history.undo(&1), Some(0));
        assert_eq!(history.undo(&0), None);
        assert_eq!((history.undo_steps(), history.redo_steps()), (0, 3));
    }

    #[test]
    fn a_new_step_clears_redo() {
        let mut history = History::new(0);
        history.observe(&1, true);
        history.observe(&2, true);
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!(history.redo_steps(), 1);
        history.observe(&5, true);
        assert_eq!(history.redo_steps(), 0);
        assert_eq!(history.redo(&5), None);
        assert_eq!(history.undo(&5), Some(1));
    }

    #[test]
    fn the_cap_drops_the_oldest_step() {
        let mut history = History::with_limit(0, 3);
        for value in 1..=5 {
            history.observe(&value, true);
        }
        assert_eq!(history.undo_steps(), 3);
        assert_eq!(history.undo(&5), Some(4));
        assert_eq!(history.undo(&4), Some(3));
        assert_eq!(history.undo(&3), Some(2));
        assert_eq!(history.undo(&2), None, "0 and 1 were dropped");
        // The full history holds 500.
        let mut history = History::new(0);
        for value in 1..=(MAX_STEPS + 20) {
            history.observe(&value, true);
        }
        assert_eq!(history.undo_steps(), MAX_STEPS);
    }

    #[test]
    fn undo_and_redo_on_empty_stacks_are_nothing() {
        let mut history = History::new(3);
        assert_eq!(history.undo(&3), None);
        assert_eq!(history.redo(&3), None);
        assert_eq!(history.present(), &3);
        assert_eq!((history.undo_steps(), history.redo_steps()), (0, 0));
    }

    #[test]
    fn an_undo_before_a_change_settled_keeps_the_change_as_a_step() {
        let mut history = History::new(0);
        history.observe(&1, true);
        // A change is pending when the undo comes.
        history.observe(&2, false);
        assert_eq!(history.undo(&2), Some(1), "back over the pending change");
        assert_eq!(history.redo(&1), Some(2), "which a redo brings back");
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!(history.undo(&1), Some(0));
    }
}
