//! The brush as a tool: arming a brush component, turning the pointer into
//! strokes, the keys and the ring cursor. The strokes themselves are data in
//! gamut-core; everything here is what the window remembers and decides.
//!
//! An armed brush owns the plain drag everywhere on the picture, the crop
//! included; Space with a drag and the middle button still pan and the wheel
//! still zooms (`view::gesture`). A stroke starts on the press with the
//! settings of that moment and ends on the release. It is in the edit from
//! its first dab, so the picture shows it and the history sees one unsettled
//! change while the pointer is down: one stroke is one undo step.
//!
//! The settings (size, feather, flow, erase) are tool state: never saved,
//! never part of the history.

use egui::{PointerButton, Pos2, Sense};
use gamut_core::brush::{Brush, MAX_BRUSH_SIZE, MIN_BRUSH_SIZE, MIN_FLOW, SharedStroke, Stroke};
use gamut_core::mask::MaskSource;

use crate::app::Session;
use crate::mask_handles::PictureMap;
use crate::view::{Gesture, Over, PanInput};

/// What one press of a bracket key multiplies or divides the size by.
pub const SIZE_STEP: f32 = 1.1;

/// What Shift with a bracket key moves the feather by.
pub const FEATHER_STEP: f32 = 5.0;

/// How far the pointer moves before a stroke takes another point, as a
/// share of the brush radius: the spacing of the dabs.
pub const POINT_SPACING: f32 = 0.25;

/// What the brush remembers between frames.
#[derive(Clone, Debug, PartialEq)]
pub struct BrushTool {
    /// The component of the selected mask that is armed: a plain drag on
    /// the picture paints into it.
    pub armed: Option<usize>,
    /// The radius as a fraction of the longer side of the photo.
    pub size: f32,
    pub feather: f32,
    pub flow: f32,
    /// The Erase toggle. Alt held does the same for one stroke.
    pub erase: bool,
    /// Where on the screen the stroke being painted took its last point.
    growing: Option<Pos2>,
    /// What Show overlay was before the brush was armed.
    overlay_before: Option<bool>,
}

impl Default for BrushTool {
    fn default() -> Self {
        let stroke = Stroke::default();
        BrushTool {
            armed: None,
            size: stroke.size,
            feather: stroke.feather,
            flow: stroke.flow,
            erase: false,
            growing: None,
            overlay_before: None,
        }
    }
}

impl BrushTool {
    /// Whether a stroke is being painted.
    pub fn is_painting(&self) -> bool {
        self.growing.is_some()
    }

    /// The settings as the stroke a press starts.
    fn stroke(&self, points: Vec<[f32; 2]>, erase: bool) -> Stroke {
        Stroke {
            points,
            size: self.size,
            feather: self.feather,
            flow: self.flow,
            erase,
        }
    }

    pub fn resize(&mut self, factor: f32) {
        self.size = (self.size * factor).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE);
    }

    pub fn soften(&mut self, by: f32) {
        self.feather = (self.feather + by).clamp(0.0, 100.0);
    }

    /// The flow kept where a stroke can hold it.
    pub fn set_flow(&mut self, flow: f32) {
        self.flow = flow.clamp(MIN_FLOW, 100.0);
    }

    /// Puts the brush down: the stroke in hand ends and `overlay` goes back
    /// to what it was before the brush was armed. Whether `overlay` changed.
    pub fn put_down(&mut self, overlay: &mut bool) -> bool {
        self.growing = None;
        let before = self.overlay_before.take();
        if self.armed.take().is_none() {
            return false;
        }
        match before {
            Some(before) if *overlay != before => {
                *overlay = before;
                true
            }
            _ => false,
        }
    }
}

/// Whether the stroke being painted takes the pointer's place as its next
/// point: once it moved a quarter of the brush radius, and never for less
/// than one pixel of the screen. `radius` is the brush radius in points.
pub fn point_due(from: Pos2, to: Pos2, radius: f32, pixels_per_point: f32) -> bool {
    let least = (radius * POINT_SPACING).max(1.0 / pixels_per_point.max(1e-6));
    (to - from).length() >= least
}

/// The points a press starts a stroke with: the place of the press, or with
/// Shift a straight line to it from where the last stroke ended. A brush
/// that holds no stroke yet has nowhere to draw a line from.
pub fn pressed_points(brush: &Brush, at: [f32; 2], shift: bool) -> Vec<[f32; 2]> {
    match brush.last_point().filter(|_| shift) {
        Some(from) => vec![from, at],
        None => vec![at],
    }
}

/// What the brush keys ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrushKey {
    Smaller,
    Larger,
    LessFeather,
    MoreFeather,
    ToggleOverlay,
    PutDown,
}

/// The brush keys as they are this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BrushKeys {
    /// A brush is armed.
    pub armed: bool,
    /// A text field has the keyboard.
    pub typing: bool,
    /// Ctrl, Cmd or Alt is held: the key belongs to something else.
    pub other_modifier: bool,
    pub shift: bool,
    /// `[` and `]`, and what Shift makes of them on a keyboard that reports
    /// the character: `{` and `}`.
    pub open: bool,
    pub close: bool,
    pub open_curly: bool,
    pub close_curly: bool,
    pub o: bool,
    pub escape: bool,
}

/// The brush keys work only while a brush is armed and nobody types. `[`
/// and `]` make the brush smaller and larger, with Shift they move the
/// feather, O shows and hides the overlay, Esc puts the brush down.
pub fn brush_key(keys: BrushKeys) -> Option<BrushKey> {
    if !keys.armed || keys.typing {
        return None;
    }
    if keys.escape {
        return Some(BrushKey::PutDown);
    }
    if keys.other_modifier {
        return None;
    }
    if keys.open_curly || (keys.shift && keys.open) {
        Some(BrushKey::LessFeather)
    } else if keys.close_curly || (keys.shift && keys.close) {
        Some(BrushKey::MoreFeather)
    } else if keys.open {
        Some(BrushKey::Smaller)
    } else if keys.close {
        Some(BrushKey::Larger)
    } else if keys.o && !keys.shift {
        Some(BrushKey::ToggleOverlay)
    } else {
        None
    }
}

/// The two rings of the brush cursor in points: the radius, and where the
/// feather starts inside it.
pub fn cursor_rings(size: f32, feather: f32, long_side: f32) -> (f32, f32) {
    let radius = size * long_side;
    (radius, radius * (1.0 - feather / 100.0))
}

impl Session {
    /// The brush the armed component holds, when the selection still has
    /// one there.
    fn armed_brush_mut(&mut self) -> Option<&mut Brush> {
        let mask = self.adjust.selected_mask?;
        let component = self.adjust.brush.armed?;
        match &mut self
            .edit
            .masks
            .get_mut(mask)?
            .components
            .get_mut(component)?
            .source
        {
            MaskSource::Brush(brush) => Some(brush),
            _ => None,
        }
    }

    /// The Paint button: arms a brush component of the selected mask, or
    /// puts the brush down when that component is the armed one. Arming
    /// turns the overlay of the mask on.
    pub fn toggle_brush(&mut self, component: usize) {
        if self.adjust.brush.armed == Some(component) {
            self.put_brush_down();
            return;
        }
        self.put_brush_down();
        self.adjust.brush.armed = Some(component);
        if self.armed_brush_mut().is_none() {
            self.adjust.brush.armed = None;
            return;
        }
        self.adjust.picking = None;
        self.adjust.brush.overlay_before = Some(self.adjust.mask_overlay);
        if !self.adjust.mask_overlay {
            self.adjust.mask_overlay = true;
            self.develop_dirty = true;
        }
    }

    /// Puts the brush down: the stroke in hand ends and the overlay goes
    /// back to what it was before the brush was armed.
    pub fn put_brush_down(&mut self) {
        if self.adjust.put_brush_down() {
            self.develop_dirty = true;
        }
    }

    /// Once a frame, and after anything that may have taken the armed
    /// component away (an undo, a deleted component, another mask selected,
    /// the Save, Discard, Cancel prompt): a brush with nothing to paint into
    /// is put down.
    pub fn check_brush(&mut self) {
        if self.adjust.brush.armed.is_none() {
            return;
        }
        if self.adjust.pending_switch.is_some() || self.armed_brush_mut().is_none() {
            self.put_brush_down();
        }
    }

    /// The press: a new stroke with the settings of this moment. `erase` is
    /// the toggle or Alt; `shift` draws a line from the end of the last
    /// stroke. `false` when no brush is armed or the brush is full.
    pub fn begin_stroke(&mut self, at: [f32; 2], screen: Pos2, erase: bool, shift: bool) -> bool {
        let tool = self.adjust.brush.clone();
        let Some(brush) = self.armed_brush_mut() else {
            return false;
        };
        if !brush.has_room() {
            self.status = Some(format!(
                "This brush holds {} strokes. Add another brush component to go on.",
                gamut_core::brush::MAX_STROKES
            ));
            return false;
        }
        let points = pressed_points(brush, at, shift);
        brush
            .strokes
            .push(SharedStroke::new(&tool.stroke(points, erase)));
        self.adjust.brush.growing = Some(screen);
        self.mark_edited();
        true
    }

    /// The drag: the stroke in hand takes the pointer's place once it is
    /// due. A stroke that is full goes on in a new one of the same brush,
    /// which the history still sees as the one change it is.
    pub fn extend_stroke(
        &mut self,
        at: [f32; 2],
        screen: Pos2,
        radius: f32,
        pixels_per_point: f32,
    ) {
        let Some(from) = self.adjust.brush.growing else {
            return;
        };
        if !point_due(from, screen, radius, pixels_per_point) {
            return;
        }
        let Some(brush) = self.armed_brush_mut() else {
            return;
        };
        let Some(stroke) = brush.strokes.last_mut() else {
            return;
        };
        if !stroke.push(at) {
            let (last, again) = (stroke.points.last().copied(), (**stroke).clone());
            if !brush.has_room() {
                return;
            }
            let points = last.into_iter().chain([at]).collect();
            brush
                .strokes
                .push(SharedStroke::new(&Stroke { points, ..again }));
        }
        self.adjust.brush.growing = Some(screen);
        self.mark_edited();
    }

    /// The release.
    pub fn end_stroke(&mut self) {
        self.adjust.brush.growing = None;
    }

    /// The Clear strokes button of a brush component of the selected mask.
    pub fn clear_strokes(&mut self, component: usize) {
        let Some(mask) = self.adjust.selected_mask else {
            return;
        };
        let source = self
            .edit
            .masks
            .get_mut(mask)
            .and_then(|mask| mask.components.get_mut(component))
            .map(|component| &mut component.source);
        if let Some(MaskSource::Brush(brush)) = source
            && !brush.strokes.is_empty()
        {
            brush.strokes.clear();
            self.adjust.brush.growing = None;
            self.mark_edited();
        }
    }

    /// Carries out a brush key.
    pub fn brush_key(&mut self, key: BrushKey) {
        match key {
            BrushKey::Smaller => self.adjust.brush.resize(1.0 / SIZE_STEP),
            BrushKey::Larger => self.adjust.brush.resize(SIZE_STEP),
            BrushKey::LessFeather => self.adjust.brush.soften(-FEATHER_STEP),
            BrushKey::MoreFeather => self.adjust.brush.soften(FEATHER_STEP),
            BrushKey::ToggleOverlay => {
                self.adjust.mask_overlay = !self.adjust.mask_overlay;
                self.develop_dirty = true;
            }
            BrushKey::PutDown => self.put_brush_down(),
        }
    }
}

/// Takes the pointer on the picture while a brush is armed and paints with
/// it. The whole tab is the brush's: over the crop and where a handle would
/// be a plain drag paints, and a drag that pans (Space, the middle button)
/// is added to `pan` and paints nothing. Every place the pointer was in this
/// frame is offered to the stroke, not only the last one, so a fast hand
/// keeps its curve.
pub fn show(ui: &mut egui::Ui, map: &PictureMap, session: &mut Session, pan: &mut PanInput) {
    session.check_brush();
    if session.adjust.brush.armed.is_none() {
        return;
    }
    let response = ui.interact(map.visible, ui.id().with("brush paint"), Sense::drag());
    let paints = pan.take(&response, Over::Brush) == Gesture::Paint;
    let (alt, shift, origin, moves) = ui.input(|i| {
        let moves: Vec<Pos2> = i
            .events
            .iter()
            .filter_map(|event| match event {
                egui::Event::PointerMoved(pos) => Some(*pos),
                _ => None,
            })
            .collect();
        (
            i.modifiers.alt,
            i.modifiers.shift,
            i.pointer.press_origin(),
            moves,
        )
    });
    let pixels_per_point = ui.pixels_per_point();
    let radius = session.adjust.brush.size * map.long_side();
    let started = response.drag_started_by(PointerButton::Primary) && paints;
    if started && let Some(origin) = origin {
        let erase = session.adjust.brush.erase || alt;
        session.begin_stroke(map.to_picture(origin), origin, erase, shift);
    }
    if session.adjust.brush.is_painting() {
        if paints {
            // On the frame of the press the moves before it are not the
            // stroke's; where the pointer is now is.
            let places = if started { Vec::new() } else { moves };
            for place in places.into_iter().chain(response.interact_pointer_pos()) {
                session.extend_stroke(map.to_picture(place), place, radius, pixels_per_point);
            }
            ui.ctx().request_repaint();
        }
        if !response.dragged_by(PointerButton::Primary) {
            session.end_stroke();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_point_is_due_after_a_quarter_radius_and_never_under_a_pixel() {
        let from = Pos2::new(100.0, 100.0);
        // A radius of 40 points: a point every 10.
        assert!(!point_due(from, Pos2::new(109.0, 100.0), 40.0, 1.0));
        assert!(point_due(from, Pos2::new(110.0, 100.0), 40.0, 1.0));
        assert!(
            point_due(from, Pos2::new(106.0, 108.0), 40.0, 1.0),
            "any direction"
        );
        // A radius of 2 points asks for half a point; a pixel is the least.
        assert!(!point_due(from, Pos2::new(100.6, 100.0), 2.0, 1.0));
        assert!(point_due(from, Pos2::new(101.0, 100.0), 2.0, 1.0));
        // On a screen of two pixels a point, a pixel is half a point.
        assert!(point_due(from, Pos2::new(100.5, 100.0), 2.0, 2.0));
        assert!(!point_due(from, Pos2::new(100.4, 100.0), 2.0, 2.0));
        assert!(
            !point_due(from, from, 40.0, 1.0),
            "a pointer that rests adds nothing"
        );
    }

    #[test]
    fn shift_draws_a_line_from_the_end_of_the_last_stroke() {
        let mut brush = Brush::default();
        assert_eq!(pressed_points(&brush, [0.5, 0.5], false), [[0.5, 0.5]]);
        assert_eq!(
            pressed_points(&brush, [0.5, 0.5], true),
            [[0.5, 0.5]],
            "nothing to draw a line from"
        );
        brush.strokes.push(SharedStroke::new(&Stroke {
            points: vec![[0.1, 0.1], [0.2, 0.3]],
            ..Stroke::default()
        }));
        assert_eq!(pressed_points(&brush, [0.5, 0.5], false), [[0.5, 0.5]]);
        assert_eq!(
            pressed_points(&brush, [0.5, 0.5], true),
            [[0.2, 0.3], [0.5, 0.5]]
        );
    }

    fn armed() -> BrushKeys {
        BrushKeys {
            armed: true,
            ..BrushKeys::default()
        }
    }

    #[test]
    fn the_brush_keys_resize_soften_toggle_and_put_down() {
        let key = |keys: BrushKeys| brush_key(keys);
        assert_eq!(
            key(BrushKeys {
                open: true,
                ..armed()
            }),
            Some(BrushKey::Smaller)
        );
        assert_eq!(
            key(BrushKeys {
                close: true,
                ..armed()
            }),
            Some(BrushKey::Larger)
        );
        let shift = BrushKeys {
            shift: true,
            ..armed()
        };
        assert_eq!(
            key(BrushKeys {
                open: true,
                ..shift
            }),
            Some(BrushKey::LessFeather)
        );
        assert_eq!(
            key(BrushKeys {
                close: true,
                ..shift
            }),
            Some(BrushKey::MoreFeather)
        );
        // A keyboard that reports the shifted character.
        assert_eq!(
            key(BrushKeys {
                open_curly: true,
                ..shift
            }),
            Some(BrushKey::LessFeather)
        );
        assert_eq!(
            key(BrushKeys {
                close_curly: true,
                ..shift
            }),
            Some(BrushKey::MoreFeather)
        );
        assert_eq!(
            key(BrushKeys { o: true, ..armed() }),
            Some(BrushKey::ToggleOverlay)
        );
        assert_eq!(
            key(BrushKeys {
                escape: true,
                ..armed()
            }),
            Some(BrushKey::PutDown)
        );
        assert_eq!(key(armed()), None);
    }

    #[test]
    fn the_brush_keys_are_silent_while_typing_and_with_no_brush_armed() {
        let every = BrushKeys {
            open: true,
            close: true,
            open_curly: true,
            close_curly: true,
            o: true,
            escape: true,
            ..BrushKeys::default()
        };
        assert_eq!(brush_key(every), None, "no brush is armed");
        assert_eq!(
            brush_key(BrushKeys {
                armed: true,
                typing: true,
                ..every
            }),
            None
        );
        // Ctrl+O opens a file and Ctrl+[ is nobody's: neither is the brush's.
        let held = BrushKeys {
            armed: true,
            other_modifier: true,
            ..BrushKeys::default()
        };
        assert_eq!(brush_key(BrushKeys { o: true, ..held }), None);
        assert_eq!(brush_key(BrushKeys { open: true, ..held }), None);
        assert_eq!(
            brush_key(BrushKeys {
                escape: true,
                ..held
            }),
            Some(BrushKey::PutDown)
        );
    }

    #[test]
    fn the_keys_hold_the_size_and_the_feather_inside_their_ranges() {
        let mut tool = BrushTool::default();
        let start = tool.size;
        tool.resize(SIZE_STEP);
        assert!((tool.size - start * 1.1).abs() < 1e-6);
        tool.resize(1.0 / SIZE_STEP);
        assert!((tool.size - start).abs() < 1e-6);
        for _ in 0..200 {
            tool.resize(SIZE_STEP);
            tool.soften(FEATHER_STEP);
        }
        assert_eq!((tool.size, tool.feather), (MAX_BRUSH_SIZE, 100.0));
        for _ in 0..400 {
            tool.resize(1.0 / SIZE_STEP);
            tool.soften(-FEATHER_STEP);
        }
        assert_eq!((tool.size, tool.feather), (MIN_BRUSH_SIZE, 0.0));
        tool.set_flow(0.0);
        assert_eq!(tool.flow, MIN_FLOW);
    }

    #[test]
    fn the_cursor_rings_are_the_radius_and_the_start_of_the_feather() {
        assert_eq!(cursor_rings(0.05, 50.0, 1000.0), (50.0, 25.0));
        assert_eq!(cursor_rings(0.05, 0.0, 1000.0), (50.0, 50.0));
        assert_eq!(cursor_rings(0.05, 100.0, 1000.0), (50.0, 0.0));
    }
}
