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
//! A pen paints with its pressure: egui hands a touch over with its force,
//! which a pen has and a finger and a mouse have not, and each point of the
//! stroke keeps the force the pen had there. One touch paints; a second one
//! ends the stroke, because two fingers are a pinch and the view owns that.
//!
//! The settings (size, feather, flow, erase, Auto mask, its sensitivity and
//! the two pressure toggles) are tool state: never saved, never part of the
//! history. A stroke keeps the settings of its press.

use egui::{Color32, CursorIcon, Key, PointerButton, Pos2, Sense, Stroke as Line, Vec2};
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
    /// Auto mask: each dab of the next stroke paints only the colour under
    /// its centre, as close as `sensitivity` asks.
    pub auto: bool,
    pub sensitivity: f32,
    /// What the pressure of a pen scales in the next stroke. A mouse and a
    /// finger paint at full pressure whatever these say.
    pub pressure_size: bool,
    pub pressure_flow: bool,
    /// The touch that paints, from its start to its end, and its force.
    touch: TouchState,
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
            auto: false,
            sensitivity: stroke.sensitivity,
            pressure_size: false,
            pressure_flow: true,
            touch: TouchState::default(),
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

    /// The settings as the stroke a press starts. `pressure` is what the
    /// pen reports at the press, and every point of the press takes it; a
    /// mouse and a finger report none and the stroke holds none.
    pub fn stroke(&self, points: Vec<[f32; 2]>, erase: bool, pressure: Option<f32>) -> Stroke {
        Stroke {
            pressure: pressure.map_or_else(Vec::new, |p| vec![p; points.len()]),
            points,
            size: self.size,
            feather: self.feather,
            flow: self.flow,
            erase,
            auto: self.auto,
            sensitivity: self.sensitivity,
            pressure_size: self.pressure_size,
            pressure_flow: self.pressure_flow,
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

/// The touch that paints. egui-winit sends a touch twice: as a touch event
/// with its force, and for the first touch also as the pointer events a
/// mouse would send. The stroke takes its points from the pointer, as it
/// does for a mouse, and only its pressure from the touch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TouchState {
    /// The touch that moves the pointer, while it is down.
    active: Option<egui::TouchId>,
    /// Its force when it last reported one: a pen. A finger reports none.
    force: Option<f32>,
}

/// What the events of one frame mean to the brush.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FrameInput {
    /// Every place the pointer was, in order, each with the pressure of the
    /// pen at that moment, or `None` from a mouse or a finger.
    pub moves: Vec<(Pos2, Option<f32>)>,
    /// The pressure at the end of the frame: what a press takes.
    pub pressure: Option<f32>,
    /// A second touch began while the first was down: two fingers are a
    /// pinch, which the view owns, and the stroke ends where it is.
    pub second_touch: bool,
}

/// Reads the events of a frame in the order the window sent them.
pub fn frame_input(events: &[egui::Event], touch: &mut TouchState) -> FrameInput {
    let mut out = FrameInput::default();
    let mut began = false;
    for event in events {
        match event {
            egui::Event::Touch {
                id, phase, force, ..
            } => match phase {
                egui::TouchPhase::Start if touch.active.is_none() => {
                    *touch = TouchState {
                        active: Some(*id),
                        force: *force,
                    };
                    began = true;
                }
                egui::TouchPhase::Start if touch.active != Some(*id) => out.second_touch = true,
                egui::TouchPhase::Start => {}
                egui::TouchPhase::Move if touch.active == Some(*id) => {
                    // A pen that lifts to no pressure reports none: the last
                    // force stands.
                    touch.force = force.or(touch.force);
                }
                egui::TouchPhase::End | egui::TouchPhase::Cancel if touch.active == Some(*id) => {
                    *touch = TouchState::default();
                }
                _ => {}
            },
            // A press that no touch of this frame began is the mouse's, and
            // a touch whose end never came counts for nothing any more.
            egui::Event::PointerButton { pressed: true, .. } if !began => {
                *touch = TouchState::default();
            }
            egui::Event::PointerMoved(pos) => {
                out.moves.push((*pos, touch.active.and(touch.force)));
            }
            _ => {}
        }
    }
    out.pressure = touch.active.and(touch.force);
    out
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
    ToggleAuto,
    PutDown,
}

/// The brush keys as they are this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BrushKeys {
    /// A brush is armed.
    pub armed: bool,
    /// A text field has the keyboard.
    pub typing: bool,
    /// Ctrl or Cmd is held: the key belongs to something else. Alt does not
    /// count, because a hand that erases holds it.
    pub other_modifier: bool,
    pub shift: bool,
    /// `[` and `]`, and what Shift makes of them on a keyboard that reports
    /// the character: `{` and `}`.
    pub open: bool,
    pub close: bool,
    pub open_curly: bool,
    pub close_curly: bool,
    pub o: bool,
    pub a: bool,
    pub escape: bool,
}

/// The brush keys work only while a brush is armed and nobody types. `[`
/// and `]` make the brush smaller and larger, with Shift they move the
/// feather, O shows and hides the overlay, A turns Auto mask on and off,
/// Esc puts the brush down.
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
    } else if keys.a && !keys.shift {
        Some(BrushKey::ToggleAuto)
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

/// Under this many points a ring is too small to see and a cross marks the
/// place instead.
const SMALLEST_RING: f32 = 2.5;

/// The brush cursor at the pointer: a ring of the brush radius at this zoom
/// and an inner ring where the feather starts, each drawn twice, dark under
/// light, so it shows on any picture. A dash in the middle says the stroke
/// will erase. With Auto mask on a small cross marks the centre: that is
/// where each dab reads its colour.
fn draw_cursor(painter: &egui::Painter, at: Pos2, rings: (f32, f32), erase: bool, auto: bool) {
    let dark = Color32::from_black_alpha(170);
    let light = Color32::from_white_alpha(235);
    let (radius, inner) = rings;
    if radius >= SMALLEST_RING {
        painter.circle_stroke(at, radius, Line::new(3.0, dark));
        painter.circle_stroke(at, radius, Line::new(1.0, light));
        if inner >= SMALLEST_RING && radius - inner >= 2.0 {
            painter.circle_stroke(at, inner, Line::new(2.0, Color32::from_black_alpha(110)));
            painter.circle_stroke(at, inner, Line::new(1.0, Color32::from_white_alpha(150)));
        }
    } else {
        for arm in [Vec2::new(6.0, 0.0), Vec2::new(0.0, 6.0)] {
            painter.line_segment([at - arm, at + arm], Line::new(3.0, dark));
            painter.line_segment([at - arm, at + arm], Line::new(1.0, light));
        }
    }
    if erase {
        let arm = Vec2::new(4.0, 0.0);
        painter.line_segment([at - arm, at + arm], Line::new(3.0, dark));
        painter.line_segment([at - arm, at + arm], Line::new(1.0, light));
    }
    // Under the smallest ring the cursor is a cross already.
    if auto && radius >= SMALLEST_RING {
        for arm in centre_cross() {
            painter.line_segment([at - arm, at + arm], Line::new(3.0, dark));
            painter.line_segment([at - arm, at + arm], Line::new(1.0, light));
        }
    }
}

/// The two arms of the cross Auto mask puts at the centre of the ring, in
/// points: shorter than the cross of a brush too small for a ring, and with
/// Erase on its flat arm is the erase dash grown a little.
pub fn centre_cross() -> [Vec2; 2] {
    [Vec2::new(5.0, 0.0), Vec2::new(0.0, 5.0)]
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
    pub fn begin_stroke(
        &mut self,
        at: [f32; 2],
        screen: Pos2,
        erase: bool,
        shift: bool,
        pressure: Option<f32>,
    ) -> bool {
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
            .push(SharedStroke::new(&tool.stroke(points, erase, pressure)));
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
        pressure: Option<f32>,
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
        if !stroke.push(at, pressure) {
            let (last, again) = (stroke.points.last().copied(), (**stroke).clone());
            if !brush.has_room() {
                return;
            }
            // The stroke goes on from its last point and that point's
            // pressure.
            let points: Vec<[f32; 2]> = last.into_iter().chain([at]).collect();
            let pressure = if again.pressure.is_empty() && pressure.is_none() {
                Vec::new()
            } else {
                let before = again.pressure.last().copied().unwrap_or(1.0);
                let held = [before, pressure.unwrap_or(before)];
                held[2 - points.len()..].to_vec()
            };
            brush.strokes.push(SharedStroke::new(&Stroke {
                points,
                pressure,
                ..again
            }));
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
            BrushKey::ToggleAuto => self.adjust.brush.auto = !self.adjust.brush.auto,
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
    // The keys first, so the ring of this frame has the size they ask for.
    // A slider keeps the focus after a click, so focus alone is not typing.
    let typing = ui.ctx().text_edit_focused();
    let keys = ui.input(|i| BrushKeys {
        armed: true,
        typing,
        other_modifier: i.modifiers.command || i.modifiers.ctrl,
        shift: i.modifiers.shift,
        open: i.key_pressed(Key::OpenBracket),
        close: i.key_pressed(Key::CloseBracket),
        open_curly: i.key_pressed(Key::OpenCurlyBracket),
        close_curly: i.key_pressed(Key::CloseCurlyBracket),
        o: i.key_pressed(Key::O),
        a: i.key_pressed(Key::A),
        escape: i.key_pressed(Key::Escape),
    });
    if let Some(key) = brush_key(keys) {
        session.brush_key(key);
        ui.ctx().request_repaint();
        if session.adjust.brush.armed.is_none() {
            return;
        }
    }
    let response = ui.interact(map.visible, ui.id().with("brush paint"), Sense::drag());
    let paints = pan.take(&response, Over::Brush) == Gesture::Paint;
    let mut touch = session.adjust.brush.touch;
    let (alt, shift, origin, input) = ui.input(|i| {
        (
            i.modifiers.alt,
            i.modifiers.shift,
            i.pointer.press_origin(),
            frame_input(&i.events, &mut touch),
        )
    });
    session.adjust.brush.touch = touch;
    let pixels_per_point = ui.pixels_per_point();
    let radius = session.adjust.brush.size * map.long_side();
    let started = response.drag_started_by(PointerButton::Primary) && paints;
    if started && let Some(origin) = origin {
        let erase = session.adjust.brush.erase || alt;
        session.begin_stroke(map.to_picture(origin), origin, erase, shift, input.pressure);
    }
    if session.adjust.brush.is_painting() {
        if paints {
            // On the frame of the press the moves before it are not the
            // stroke's; where the pointer is now is.
            let places = if started { Vec::new() } else { input.moves };
            let now = response
                .interact_pointer_pos()
                .map(|place| (place, input.pressure));
            for (place, pressure) in places.into_iter().chain(now) {
                let at = map.to_picture(place);
                session.extend_stroke(at, place, radius, pixels_per_point, pressure);
            }
            ui.ctx().request_repaint();
        }
        if !response.dragged_by(PointerButton::Primary) || input.second_touch {
            session.end_stroke();
        }
    }
    // The ring in place of the system pointer, over the picture only. With
    // Space held or a pan under way the viewer shows the hand instead.
    let pointer = ui
        .input(|i| i.pointer.hover_pos())
        .filter(|_| response.hovered() || response.dragged());
    if let Some(at) = pointer
        && !pan.space
        && !pan.panning
    {
        ui.ctx().set_cursor_icon(CursorIcon::None);
        let tool = &session.adjust.brush;
        let rings = cursor_rings(tool.size, tool.feather, map.long_side());
        let painter = ui.painter().with_clip_rect(map.visible);
        draw_cursor(&painter, at, rings, tool.erase || alt, tool.auto);
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
            key(BrushKeys { a: true, ..armed() }),
            Some(BrushKey::ToggleAuto)
        );
        assert_eq!(
            key(BrushKeys { a: true, ..shift }),
            None,
            "Shift+A is not the brush's"
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
            a: true,
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
        // Ctrl+A selects all in a text field.
        assert_eq!(brush_key(BrushKeys { a: true, ..held }), None);
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
    fn a_press_stores_the_settings_of_the_tool_in_the_stroke() {
        let tool = BrushTool::default();
        assert!(!tool.auto && !tool.pressure_size, "off until asked for");
        assert!(
            tool.pressure_flow,
            "a pen paints lighter with a lighter hand"
        );
        let plain = tool.stroke(vec![[0.5, 0.5]], false, None);
        assert!(!plain.auto && plain.pressure.is_empty());
        assert!(plain.pressure_flow && !plain.pressure_size);

        let tool = BrushTool {
            auto: true,
            sensitivity: 80.0,
            pressure_size: true,
            pressure_flow: false,
            ..BrushTool::default()
        };
        let line = tool.stroke(vec![[0.1, 0.1], [0.5, 0.5]], true, Some(0.4));
        assert!(line.auto && line.erase);
        assert_eq!(line.sensitivity, 80.0);
        assert!(line.pressure_size && !line.pressure_flow);
        assert_eq!(
            line.pressure,
            [0.4, 0.4],
            "a Shift line is at the pressure of the press"
        );
        assert_eq!(SharedStroke::new(&line).pressure.len(), line.points.len());
    }

    fn touch(id: u64, phase: egui::TouchPhase, pos: Pos2, force: Option<f32>) -> egui::Event {
        egui::Event::Touch {
            device_id: egui::TouchDeviceId(1),
            id: egui::TouchId(id),
            phase,
            pos,
            force,
        }
    }

    fn press(pos: Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        }
    }

    #[test]
    fn a_point_takes_the_force_of_a_pen_and_none_from_a_finger_or_a_mouse() {
        use egui::Event::PointerMoved;
        use egui::TouchPhase::{End, Move, Start};
        let (a, b, c) = (
            Pos2::new(10.0, 10.0),
            Pos2::new(20.0, 10.0),
            Pos2::new(30.0, 10.0),
        );

        // A pen, as egui-winit sends it: the touch first, then the pointer.
        let mut pen = TouchState::default();
        let down = frame_input(
            &[
                touch(7, Start, a, Some(0.2)),
                PointerMoved(a),
                press(a, true),
            ],
            &mut pen,
        );
        assert_eq!(down.moves, [(a, Some(0.2))]);
        assert_eq!(down.pressure, Some(0.2));
        let moved = frame_input(
            &[
                touch(7, Move, b, Some(0.5)),
                PointerMoved(b),
                touch(7, Move, c, Some(0.9)),
                PointerMoved(c),
            ],
            &mut pen,
        );
        assert_eq!(
            moved.moves,
            [(b, Some(0.5)), (c, Some(0.9))],
            "each move its own force"
        );
        // A pen that reports no force for a moment keeps the last one.
        let held = frame_input(&[touch(7, Move, c, None), PointerMoved(c)], &mut pen);
        assert_eq!(held.moves, [(c, Some(0.9))]);
        let up = frame_input(&[touch(7, End, c, None), press(c, false)], &mut pen);
        assert_eq!(up.pressure, None, "lifted");
        assert_eq!(pen, TouchState::default());

        // A finger reports no force.
        let mut finger = TouchState::default();
        let down = frame_input(
            &[touch(3, Start, a, None), PointerMoved(a), press(a, true)],
            &mut finger,
        );
        assert_eq!((down.moves, down.pressure), (vec![(a, None)], None));

        // A mouse sends no touch at all, and a press of its own drops a touch
        // whose end never came.
        let mut stale = TouchState::default();
        frame_input(&[touch(7, Start, a, Some(0.3))], &mut stale);
        let mouse = frame_input(
            &[PointerMoved(b), press(b, true), PointerMoved(c)],
            &mut stale,
        );
        assert_eq!(mouse.moves, [(b, Some(0.3)), (c, None)]);
        assert_eq!(mouse.pressure, None);
    }

    #[test]
    fn a_second_touch_is_told_from_the_one_that_paints() {
        use egui::TouchPhase::{Cancel, Move, Start};
        let at = Pos2::new(10.0, 10.0);
        let mut state = TouchState::default();
        assert!(!frame_input(&[touch(1, Start, at, None)], &mut state).second_touch);
        assert!(!frame_input(&[touch(1, Move, at, None)], &mut state).second_touch);
        assert!(frame_input(&[touch(2, Start, at, None)], &mut state).second_touch);
        // The second finger moving or lifting changes nothing of the first.
        let before = state;
        frame_input(
            &[touch(2, Move, at, Some(0.5)), touch(2, Cancel, at, None)],
            &mut state,
        );
        assert_eq!(state, before);
        frame_input(&[touch(1, Cancel, at, None)], &mut state);
        assert_eq!(state, TouchState::default());
    }

    #[test]
    fn the_cross_of_auto_mask_is_smaller_than_the_cross_of_a_tiny_brush() {
        for arm in centre_cross() {
            assert!(arm.length() < 6.0 && arm.length() > 4.0);
        }
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
    fn the_ring_is_the_brush_radius_on_the_screen_at_fit_at_100_and_at_400_percent() {
        use crate::view::{View, Zoom};
        use egui::Rect;
        let tab = Rect::from_min_size(Pos2::new(40.0, 30.0), Vec2::new(1200.0, 900.0));
        let source = (6000, 4000);
        let size = 0.05;
        // Fitted, the longer side of the photo is the width of the tab; at a
        // scale it is that many screen pixels for every pixel of the photo.
        for (zoom, long_side) in [
            (Zoom::Fit, 1200.0),
            (Zoom::Scale(1.0), 6000.0),
            (Zoom::Scale(4.0), 24000.0),
        ] {
            let view = View {
                zoom,
                centre: [0.4, 0.6],
            };
            let map = PictureMap::new(&view.place(tab, source, 1.0), tab);
            assert!(
                (map.long_side() - long_side).abs() < 0.5,
                "{zoom:?}: {}",
                map.long_side()
            );
            let (radius, inner) = cursor_rings(size, 40.0, map.long_side());
            assert!(
                (radius - size * long_side).abs() < 0.05,
                "{zoom:?}: {radius}"
            );
            assert!((inner - radius * 0.6).abs() < 0.05);
            // What the ring shows is what gets painted: a point on the ring
            // is one brush radius from the centre on the photo.
            let centre = map.to_screen([0.4, 0.6]);
            let on_ring = map.to_picture(centre + Vec2::new(radius, 0.0));
            assert!(
                (on_ring[0] - 0.4 - size).abs() < 1e-4,
                "{zoom:?}: {on_ring:?}"
            );
        }
        // On a screen of two pixels a point the ring is half as many points.
        let view = View {
            zoom: Zoom::Scale(1.0),
            centre: [0.5, 0.5],
        };
        let map = PictureMap::new(&view.place(tab, source, 2.0), tab);
        assert!((cursor_rings(size, 0.0, map.long_side()).0 - 150.0).abs() < 0.05);
    }

    #[test]
    fn the_cursor_rings_are_the_radius_and_the_start_of_the_feather() {
        assert_eq!(cursor_rings(0.05, 50.0, 1000.0), (50.0, 25.0));
        assert_eq!(cursor_rings(0.05, 0.0, 1000.0), (50.0, 50.0));
        assert_eq!(cursor_rings(0.05, 100.0, 1000.0), (50.0, 0.0));
    }
}
