//! The view of the open picture: how far it is zoomed and which part of it
//! sits in the middle of the Viewer tab. A view belongs to the open file and
//! to the window; it is never saved.
//!
//! A scale is output pixels per source pixel, so 1.0 is 100 percent. The
//! least scale is the one that fits the whole picture into the tab and the
//! view at that scale is [`Zoom::Fit`], which keeps fitting when the tab is
//! resized. Everything here is arithmetic on rectangles: the viewer asks
//! where the picture goes ([`View::place`]) and what to render for it
//! ([`RenderPlan`]), and the pointer code asks for a view that keeps a point
//! still ([`View::zoomed_about`]).
//!
//! A zoomed view never renders the whole picture. It renders a window of
//! the source: what is seen, padded by half a tab on every side and snapped
//! to a grid, so that a pan inside the window moves the output crop only.
//! The render never passes one source pixel per output pixel, because above
//! that the blur radius meets its cap and the look would change with the
//! zoom; past 100 percent the 100 percent render is magnified.

use egui::{Pos2, Rect, Vec2};

pub use gamut_gpu::develop::{PixelRect, holds, padded_window};

use crate::viewer::fit_aspect;

/// The most a picture is magnified: 800 percent, unless fitting the tab
/// already takes more.
pub const MAX_SCALE: f32 = 8.0;

/// One notch of the wheel, and one press of the zoom keys.
pub const ZOOM_STEP: f32 = 1.25;

/// A scale this close to the fit scale is the fit scale.
const FIT_SNAP: f32 = 1e-3;

/// The grid the padded window is snapped to, in source pixels.
pub const WINDOW_GRID: u32 = 64;

/// From this magnification single pixels are shown as squares; under it
/// the magnified render is filtered.
pub const NEAREST_FROM: f32 = 4.0;

/// How far the picture is zoomed.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Zoom {
    /// The whole picture, as large as the tab allows.
    #[default]
    Fit,
    /// Output pixels per source pixel.
    Scale(f32),
}

/// The zoom and the point of the picture in the middle of the tab,
/// normalised to the uncropped picture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub zoom: Zoom,
    pub centre: [f32; 2],
}

impl Default for View {
    fn default() -> Self {
        View {
            zoom: Zoom::Fit,
            centre: [0.5, 0.5],
        }
    }
}

/// Where a view puts the picture in a tab.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// The rectangle the whole picture occupies, in points. It is larger
    /// than the tab when the view is zoomed in.
    pub image: Rect,
    /// Output pixels per source pixel.
    pub scale: f32,
    /// The view after its clamps: what the next frame starts from.
    pub view: View,
}

/// The scale at which the whole picture fits the tab.
pub fn fit_scale(tab: Vec2, source: (u32, u32), pixels_per_point: f32) -> f32 {
    let (width, height) = (source.0.max(1) as f32, source.1.max(1) as f32);
    let fitted = fit_aspect(tab, [width, height]);
    (fitted.x * pixels_per_point / width).max(1e-6)
}

/// The most the picture is magnified in this tab.
pub fn max_scale(fit: f32) -> f32 {
    MAX_SCALE.max(fit)
}

impl View {
    /// The scale this view shows at.
    pub fn scale(&self, fit: f32) -> f32 {
        match self.zoom {
            Zoom::Fit => fit,
            Zoom::Scale(scale) => scale.clamp(fit, max_scale(fit)),
        }
    }

    /// The view with its scale inside the limits, snapped to Fit at the fit
    /// scale, and its centre held so that a picture larger than the tab
    /// covers it and a smaller one is centred.
    pub fn clamped(&self, tab: Vec2, source: (u32, u32), pixels_per_point: f32) -> View {
        let fit = fit_scale(tab, source, pixels_per_point);
        let scale = self.scale(fit);
        if scale <= fit * (1.0 + FIT_SNAP) {
            return View::default();
        }
        let size = picture_size(source, scale, pixels_per_point);
        let axis = |centre: f32, picture: f32, tab: f32| {
            if picture <= tab {
                0.5
            } else {
                let half = tab / (2.0 * picture);
                centre.clamp(half, 1.0 - half)
            }
        };
        View {
            zoom: Zoom::Scale(scale),
            centre: [
                axis(self.centre[0], size.x, tab.x),
                axis(self.centre[1], size.y, tab.y),
            ],
        }
    }

    /// Where the picture goes in `tab`. A Fit view is placed exactly as the
    /// viewer placed the picture before it could zoom. A zoomed picture
    /// starts on a whole pixel of the screen, so that at 100 percent every
    /// source pixel is one screen pixel and nothing is filtered.
    pub fn place(&self, tab: Rect, source: (u32, u32), pixels_per_point: f32) -> Placement {
        let mut placed = self.place_exact(tab, source, pixels_per_point);
        if placed.view.zoom != Zoom::Fit {
            let ppp = pixels_per_point.max(1e-6);
            let min = Pos2::new(
                (placed.image.min.x * ppp).round() / ppp,
                (placed.image.min.y * ppp).round() / ppp,
            );
            placed.image = Rect::from_min_size(min, placed.image.size());
        }
        placed
    }

    /// [`place`](Self::place) before the snap to the pixels of the screen:
    /// what the zoom and the pan are worked out on.
    pub fn place_exact(&self, tab: Rect, source: (u32, u32), pixels_per_point: f32) -> Placement {
        let view = self.clamped(tab.size(), source, pixels_per_point);
        match view.zoom {
            Zoom::Fit => {
                let aspect = [source.0.max(1) as f32, source.1.max(1) as f32];
                Placement {
                    image: Rect::from_center_size(tab.center(), fit_aspect(tab.size(), aspect)),
                    scale: fit_scale(tab.size(), source, pixels_per_point),
                    view,
                }
            }
            Zoom::Scale(scale) => {
                let size = picture_size(source, scale, pixels_per_point);
                let min =
                    tab.center() - Vec2::new(view.centre[0] * size.x, view.centre[1] * size.y);
                Placement {
                    image: Rect::from_min_size(min, size),
                    scale,
                    view,
                }
            }
        }
    }

    /// The view at `scale` that keeps the point of the picture under
    /// `anchor` where it is, as far as the clamps allow.
    pub fn zoomed_about(
        &self,
        tab: Rect,
        source: (u32, u32),
        pixels_per_point: f32,
        anchor: Pos2,
        scale: f32,
    ) -> View {
        let placed = self.place_exact(tab, source, pixels_per_point);
        let fit = fit_scale(tab.size(), source, pixels_per_point);
        let scale = scale.clamp(fit, max_scale(fit));
        let under = [
            (anchor.x - placed.image.min.x) / placed.image.width().max(1e-6),
            (anchor.y - placed.image.min.y) / placed.image.height().max(1e-6),
        ];
        let size = picture_size(source, scale, pixels_per_point);
        let offset = anchor - tab.center();
        View {
            zoom: Zoom::Scale(scale),
            centre: [
                under[0] - offset.x / size.x.max(1e-6),
                under[1] - offset.y / size.y.max(1e-6),
            ],
        }
        .clamped(tab.size(), source, pixels_per_point)
    }

    /// One step in (`steps` above 0) or out about `anchor`.
    pub fn stepped(
        &self,
        tab: Rect,
        source: (u32, u32),
        pixels_per_point: f32,
        anchor: Pos2,
        steps: f32,
    ) -> View {
        let placed = self.place_exact(tab, source, pixels_per_point);
        let scale = placed.scale * ZOOM_STEP.powf(steps);
        self.zoomed_about(tab, source, pixels_per_point, anchor, scale)
    }

    /// The view moved by a drag of `delta` points: the picture follows the
    /// pointer.
    pub fn panned(
        &self,
        tab: Rect,
        source: (u32, u32),
        pixels_per_point: f32,
        delta: Vec2,
    ) -> View {
        let placed = self.place_exact(tab, source, pixels_per_point);
        if placed.view.zoom == Zoom::Fit {
            return placed.view;
        }
        View {
            zoom: placed.view.zoom,
            centre: [
                placed.view.centre[0] - delta.x / placed.image.width().max(1e-6),
                placed.view.centre[1] - delta.y / placed.image.height().max(1e-6),
            ],
        }
        .clamped(tab.size(), source, pixels_per_point)
    }
}

/// What a drag on the picture is over when it starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Over {
    /// A handle of the selected mask.
    Handle,
    /// The crop rectangle.
    Crop,
    /// The picture, or the tab around it.
    Picture,
    /// Anywhere on the picture while a brush is armed: the brush owns the
    /// plain drag, over the crop and where a handle would be too.
    Brush,
}

/// What a drag starts over: whatever is under the pointer, unless a brush is
/// armed, which takes the place of all of it.
pub fn over(brush_armed: bool, under: Over) -> Over {
    if brush_armed { Over::Brush } else { under }
}

/// What a drag on the picture does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gesture {
    Pan,
    HandleDrag,
    CropDrag,
    /// A stroke of the armed brush.
    Paint,
    Nothing,
}

/// What a pointer and key state means. The middle button pans wherever it
/// starts, and so does the primary button while Space is held. A plain drag
/// does what it did before the viewer could zoom: it moves the handle or the
/// crop it starts on, and nothing on the bare picture. With a brush armed it
/// paints.
pub fn gesture(over: Over, primary: bool, middle: bool, space: bool) -> Gesture {
    if middle || (primary && space) {
        Gesture::Pan
    } else if primary {
        match over {
            Over::Handle => Gesture::HandleDrag,
            Over::Crop => Gesture::CropDrag,
            Over::Picture => Gesture::Nothing,
            Over::Brush => Gesture::Paint,
        }
    } else {
        Gesture::Nothing
    }
}

/// The pan the widgets on the picture collect during a frame: every one of
/// them can be the start of a pan, so each reports its drag here.
#[derive(Clone, Copy, Debug, Default)]
pub struct PanInput {
    /// Space is held and no text field has the keyboard.
    pub space: bool,
    /// How far the pans of this frame moved, in points.
    pub delta: Vec2,
    /// A pan is under way, moving or not.
    pub panning: bool,
}

impl PanInput {
    /// What the drag on `response` means, with a pan added to the total.
    pub fn take(&mut self, response: &egui::Response, over: Over) -> Gesture {
        let found = gesture(
            over,
            response.dragged_by(egui::PointerButton::Primary),
            response.dragged_by(egui::PointerButton::Middle),
            self.space,
        );
        if found == Gesture::Pan {
            self.delta += response.drag_delta();
            self.panning = true;
        }
        found
    }
}

/// A change of view a key or a button of the Adjust tab asks for. The
/// viewer carries it out, because only it knows the tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewKey {
    /// Ctrl+0 and F: the whole picture.
    Fit,
    /// Ctrl+1: 100 percent about the middle of the tab.
    Actual,
    /// Ctrl+= and Ctrl+-: one step about the middle of the tab.
    In,
    Out,
}

/// The keys of the view as they are this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewKeys {
    /// Ctrl, or Cmd on a Mac, with nothing else but Shift (which `+` needs
    /// on some layouts).
    pub command: bool,
    /// No modifier at all.
    pub bare: bool,
    pub zero: bool,
    pub one: bool,
    pub plus: bool,
    pub minus: bool,
    pub f: bool,
    /// A text field or a drag value has the keyboard.
    pub typing: bool,
}

/// The change of view the keys ask for, none while someone types.
pub fn view_key(keys: ViewKeys) -> Option<ViewKey> {
    if keys.typing {
        return None;
    }
    if keys.command {
        if keys.zero {
            return Some(ViewKey::Fit);
        }
        if keys.one {
            return Some(ViewKey::Actual);
        }
        if keys.plus {
            return Some(ViewKey::In);
        }
        if keys.minus {
            return Some(ViewKey::Out);
        }
    }
    (keys.bare && keys.f).then_some(ViewKey::Fit)
}

/// Points of touchpad scroll that count as one notch of a wheel.
const POINTS_PER_NOTCH: f32 = 40.0;

/// The zoom steps of one wheel event: a notch of a wheel is one step, a
/// touchpad scrolls [`POINTS_PER_NOTCH`] points for one.
pub fn wheel_steps(unit: egui::MouseWheelUnit, delta: Vec2) -> f32 {
    match unit {
        egui::MouseWheelUnit::Point => delta.y / POINTS_PER_NOTCH,
        egui::MouseWheelUnit::Line | egui::MouseWheelUnit::Page => delta.y,
    }
}

/// Whether the release of Space plays or pauses: only a Space that panned
/// nothing while it was held, and never one typed into a text field.
pub fn space_release_toggles(panned: bool, typing: bool) -> bool {
    !panned && !typing
}

/// What the Viewer and the other tabs tell each other about the view.
#[derive(Clone, Copy, Debug, Default)]
pub struct ViewLink {
    /// A change of view the Adjust tab asks the viewer for.
    pub request: Option<ViewKey>,
    /// The scale the viewer last showed and whether that was Fit; `None`
    /// until a picture has been shown.
    pub shown: Option<(f32, bool)>,
    /// The Space that is held has panned, so its release toggles nothing.
    pub space_panned: bool,
}

impl View {
    /// The view after a key or a button, about the middle of the tab.
    pub fn after_key(
        &self,
        key: ViewKey,
        tab: Rect,
        source: (u32, u32),
        pixels_per_point: f32,
    ) -> View {
        let middle = tab.center();
        match key {
            ViewKey::Fit => View::default(),
            ViewKey::Actual => self.zoomed_about(tab, source, pixels_per_point, middle, 1.0),
            ViewKey::In => self.stepped(tab, source, pixels_per_point, middle, 1.0),
            ViewKey::Out => self.stepped(tab, source, pixels_per_point, middle, -1.0),
        }
    }
}

/// What a zoomed view renders, in pixels of the whole picture at the render
/// scale, and where the result is painted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPlan {
    /// The size of the whole picture at the render scale: the view's scale,
    /// or the size of the source from 100 percent up.
    pub full: (u32, u32),
    /// The part of `full` that is seen, on whole pixels. Its size depends on
    /// the tab and the zoom only, so a pan never resizes the output.
    pub visible: PixelRect,
    /// Half a tab, the padding of the window on every side.
    pub pad: (u32, u32),
    /// [`WINDOW_GRID`] at the render scale.
    pub grid: u32,
    /// Where the rendered `visible` goes on the screen, in points.
    pub paint: Rect,
    /// Whether the magnified render shows its pixels as squares.
    pub nearest: bool,
}

impl RenderPlan {
    /// The plan of a placed view, or `None` for a Fit view, which renders
    /// the whole picture at the fitted size as it always did.
    pub fn of(
        placed: &Placement,
        tab: Rect,
        source: (u32, u32),
        pixels_per_point: f32,
    ) -> Option<RenderPlan> {
        if placed.view.zoom == Zoom::Fit {
            return None;
        }
        let render_scale = placed.scale.min(1.0);
        let full = (
            ((source.0.max(1) as f32 * render_scale).round() as u32).clamp(1, source.0.max(1)),
            ((source.1.max(1) as f32 * render_scale).round() as u32).clamp(1, source.1.max(1)),
        );
        let image = placed.image;
        let seen = tab.intersect(image);
        // Pixels of `full` per point of the screen, on each axis.
        let per_point = Vec2::new(
            full.0 as f32 / image.width().max(1e-6),
            full.1 as f32 / image.height().max(1e-6),
        );
        let axis = |from: f32, extent: f32, per_point: f32, full: u32| {
            let size = ((extent * per_point).ceil() as u32 + 1).clamp(1, full);
            let start = ((from * per_point).floor().max(0.0) as u32).min(full - size);
            (start, size)
        };
        let (x, width) = axis(seen.min.x - image.min.x, seen.width(), per_point.x, full.0);
        let (y, height) = axis(seen.min.y - image.min.y, seen.height(), per_point.y, full.1);
        let tab_pixels = tab.size() * pixels_per_point * (render_scale / placed.scale.max(1e-6));
        Some(RenderPlan {
            full,
            visible: (x, y, width, height),
            pad: (
                (tab_pixels.x / 2.0).ceil() as u32,
                (tab_pixels.y / 2.0).ceil() as u32,
            ),
            grid: ((WINDOW_GRID as f32 * render_scale).round() as u32).max(1),
            paint: Rect::from_min_size(
                image.min + Vec2::new(x as f32 / per_point.x, y as f32 / per_point.y),
                Vec2::new(width as f32 / per_point.x, height as f32 / per_point.y),
            ),
            nearest: placed.scale >= NEAREST_FROM,
        })
    }

    /// The window to render for this plan: the one already rendered while
    /// it is of the same picture size and still holds what is seen, a new
    /// padded one otherwise. From 100 percent up the picture size no longer
    /// changes, so zooming further in keeps the window too.
    pub fn window(&self, rendered: Option<((u32, u32), PixelRect)>) -> PixelRect {
        match rendered {
            Some((full, window)) if full == self.full && holds(window, self.visible) => window,
            _ => padded_window(self.full, self.visible, self.pad, self.grid),
        }
    }
}

/// The size of the whole picture in points at a scale.
fn picture_size(source: (u32, u32), scale: f32, pixels_per_point: f32) -> Vec2 {
    Vec2::new(source.0.max(1) as f32, source.1.max(1) as f32) * (scale / pixels_per_point.max(1e-6))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE: (u32, u32) = (6000, 4000);

    fn tab() -> Rect {
        Rect::from_min_size(Pos2::new(40.0, 30.0), Vec2::new(900.0, 700.0))
    }

    /// The point of the picture under a screen position.
    fn under(view: &View, at: Pos2, pixels_per_point: f32) -> [f32; 2] {
        let image = view.place_exact(tab(), SOURCE, pixels_per_point).image;
        [
            (at.x - image.min.x) / image.width(),
            (at.y - image.min.y) / image.height(),
        ]
    }

    fn close(a: [f32; 2], b: [f32; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4
    }

    #[test]
    fn the_fit_scale_is_the_side_that_limits() {
        // 900 by 700 points holds a 3:2 picture 900 wide: 0.15 at one pixel
        // per point, twice that on a screen of two.
        assert!((fit_scale(tab().size(), SOURCE, 1.0) - 0.15).abs() < 1e-6);
        assert!((fit_scale(tab().size(), SOURCE, 2.0) - 0.30).abs() < 1e-6);
        // A tall picture is limited by the height of the tab.
        assert!((fit_scale(tab().size(), (4000, 7000), 1.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn a_fit_view_is_placed_as_the_viewer_placed_it_before_zoom() {
        let placed = View::default().place(tab(), SOURCE, 1.0);
        let before =
            Rect::from_center_size(tab().center(), fit_aspect(tab().size(), [6000.0, 4000.0]));
        assert_eq!(placed.image, before);
        assert_eq!(placed.view, View::default());
        assert_eq!(placed.image.size(), Vec2::new(900.0, 600.0));
    }

    #[test]
    fn the_scale_is_held_between_fit_and_eight() {
        let fit = fit_scale(tab().size(), SOURCE, 1.0);
        let far = View {
            zoom: Zoom::Scale(40.0),
            centre: [0.5, 0.5],
        };
        assert_eq!(far.place(tab(), SOURCE, 1.0).scale, MAX_SCALE);
        let near = View {
            zoom: Zoom::Scale(fit / 3.0),
            centre: [0.5, 0.5],
        };
        assert_eq!(near.place(tab(), SOURCE, 1.0).view, View::default());
        // A picture so small that fitting magnifies it past 800 percent
        // stops at the fit scale.
        let tiny = (40, 30);
        let fit = fit_scale(tab().size(), tiny, 1.0);
        assert!(fit > MAX_SCALE);
        assert_eq!(max_scale(fit), fit);
        let view = View::default().stepped(tab(), tiny, 1.0, tab().center(), 1.0);
        assert_eq!(view, View::default());
    }

    #[test]
    fn a_wheel_step_in_keeps_the_point_under_the_pointer() {
        // Near enough to the middle that the first step, which barely
        // outgrows the tab, is not held by the clamp.
        let pointer = Pos2::new(610.0, 360.0);
        for pixels_per_point in [1.0, 1.5] {
            let mut view = View::default();
            for _ in 0..6 {
                let before = under(&view, pointer, pixels_per_point);
                view = view.stepped(tab(), SOURCE, pixels_per_point, pointer, 1.0);
                assert!(matches!(view.zoom, Zoom::Scale(_)));
                assert!(close(before, under(&view, pointer, pixels_per_point)));
            }
        }
    }

    #[test]
    fn a_wheel_step_out_keeps_the_point_under_the_pointer() {
        let pointer = Pos2::new(500.0, 400.0);
        let mut view = View {
            zoom: Zoom::Scale(2.0),
            centre: [0.5, 0.5],
        };
        for _ in 0..4 {
            let before = under(&view, pointer, 1.0);
            view = view.stepped(tab(), SOURCE, 1.0, pointer, -1.0);
            assert!(close(before, under(&view, pointer, 1.0)));
        }
    }

    #[test]
    fn a_jump_to_100_percent_keeps_the_point_under_the_pointer() {
        let pointer = Pos2::new(300.0, 500.0);
        let view = View::default();
        let before = under(&view, pointer, 1.0);
        let view = view.zoomed_about(tab(), SOURCE, 1.0, pointer, 1.0);
        assert_eq!(view.zoom, Zoom::Scale(1.0));
        assert!(close(before, under(&view, pointer, 1.0)));
    }

    #[test]
    fn zooming_out_to_the_fit_scale_snaps_to_fit() {
        let mut view = View::default().stepped(tab(), SOURCE, 1.0, Pos2::new(200.0, 200.0), 2.0);
        assert!(matches!(view.zoom, Zoom::Scale(_)));
        for _ in 0..3 {
            view = view.stepped(tab(), SOURCE, 1.0, Pos2::new(700.0, 600.0), -1.0);
        }
        assert_eq!(view, View::default());
    }

    #[test]
    fn a_larger_picture_always_covers_the_tab() {
        for centre in [[-3.0, -3.0], [0.0, 1.0], [9.0, 0.2]] {
            let view = View {
                zoom: Zoom::Scale(1.0),
                centre,
            };
            let image = view.place(tab(), SOURCE, 1.0).image;
            assert!(image.min.x <= tab().min.x + 1e-3 && image.max.x >= tab().max.x - 1e-3);
            assert!(image.min.y <= tab().min.y + 1e-3 && image.max.y >= tab().max.y - 1e-3);
        }
    }

    #[test]
    fn a_side_smaller_than_the_tab_stays_centred() {
        // At 0.16 the 3:2 picture is 960 by 640 points: wider than the tab,
        // not as tall.
        let view = View {
            zoom: Zoom::Scale(0.16),
            centre: [0.9, 0.9],
        };
        let placed = view.place(tab(), SOURCE, 1.0);
        assert_eq!(placed.view.centre[1], 0.5);
        assert!((placed.image.center().y - tab().center().y).abs() < 1e-3);
        assert!((placed.image.max.x - tab().max.x).abs() < 1e-3);
    }

    #[test]
    fn a_resized_tab_keeps_a_fit_view_fitting() {
        let view = View::default();
        for size in [Vec2::new(300.0, 900.0), Vec2::new(1600.0, 400.0)] {
            let tab = Rect::from_min_size(Pos2::ZERO, size);
            let placed = view.place(tab, SOURCE, 1.0);
            assert_eq!(placed.view, View::default());
            assert_eq!(placed.image.size(), fit_aspect(size, [6000.0, 4000.0]));
        }
        // A zoomed view a larger tab now fits becomes Fit.
        let zoomed = View {
            zoom: Zoom::Scale(0.2),
            centre: [0.3, 0.3],
        };
        let large = Rect::from_min_size(Pos2::ZERO, Vec2::new(3000.0, 2000.0));
        assert_eq!(zoomed.place(large, SOURCE, 1.0).view, View::default());
    }

    #[test]
    fn a_pan_moves_the_picture_with_the_pointer_and_stops_at_its_edge() {
        let view = View {
            zoom: Zoom::Scale(1.0),
            centre: [0.5, 0.5],
        };
        let before = view.place_exact(tab(), SOURCE, 1.0).image;
        let panned = view.panned(tab(), SOURCE, 1.0, Vec2::new(120.0, -80.0));
        let after = panned.place_exact(tab(), SOURCE, 1.0).image;
        assert!((after.min - before.min - Vec2::new(120.0, -80.0)).length() < 1e-2);
        let far = view.panned(tab(), SOURCE, 1.0, Vec2::new(1e6, 1e6));
        let image = far.place_exact(tab(), SOURCE, 1.0).image;
        assert!((image.min - tab().min).length() < 1e-2);
        // A fitted picture has nowhere to go.
        assert_eq!(
            View::default().panned(tab(), SOURCE, 1.0, Vec2::new(50.0, 50.0)),
            View::default()
        );
    }

    fn at(scale: f32, centre: [f32; 2]) -> View {
        View {
            zoom: Zoom::Scale(scale),
            centre,
        }
    }

    fn plan(view: &View, pixels_per_point: f32) -> RenderPlan {
        let placed = view.place(tab(), SOURCE, pixels_per_point);
        RenderPlan::of(&placed, tab(), SOURCE, pixels_per_point).expect("a zoomed view")
    }

    #[test]
    fn a_zoomed_picture_starts_on_a_whole_pixel_of_the_screen() {
        for pixels_per_point in [1.0, 1.5, 2.0] {
            let view = at(1.0, [0.31337, 0.64123]);
            let exact = view.place_exact(tab(), SOURCE, pixels_per_point).image;
            let placed = view.place(tab(), SOURCE, pixels_per_point).image;
            let min = placed.min.to_vec2() * pixels_per_point;
            assert!((min.x - min.x.round()).abs() < 1e-3 && (min.y - min.y.round()).abs() < 1e-3);
            // Never more than half a pixel from where the arithmetic put it.
            assert!(
                ((placed.min - exact.min) * pixels_per_point)
                    .abs()
                    .max_elem()
                    <= 0.5 + 1e-3
            );
            assert_eq!(placed.size(), exact.size());
        }
        // A fitted picture is where it always was.
        let fit = View::default();
        assert_eq!(
            fit.place(tab(), SOURCE, 1.5).image,
            fit.place_exact(tab(), SOURCE, 1.5).image
        );
    }

    #[test]
    fn a_fit_view_has_no_plan_and_renders_as_before() {
        let placed = View::default().place(tab(), SOURCE, 1.0);
        assert_eq!(RenderPlan::of(&placed, tab(), SOURCE, 1.0), None);
    }

    #[test]
    fn the_render_never_passes_the_resolution_of_the_source() {
        for scale in [0.2, 0.5, 1.0, 2.0, 4.0, 8.0] {
            let plan = plan(&at(scale, [0.4, 0.6]), 1.0);
            assert!(
                plan.full.0 <= SOURCE.0 && plan.full.1 <= SOURCE.1,
                "{scale}"
            );
            if scale >= 1.0 {
                assert_eq!(plan.full, SOURCE, "from 100 percent up it is the source");
            }
            assert_eq!(plan.nearest, scale >= NEAREST_FROM);
        }
        assert_eq!(plan(&at(0.5, [0.5, 0.5]), 1.0).full, (3000, 2000));
    }

    #[test]
    fn what_is_seen_covers_the_tab_on_whole_pixels() {
        // At 100 percent the tab of 900 by 700 sees 900 by 700 source pixels,
        // and one more so the last partial pixel is covered.
        let view = at(1.0, [0.4, 0.6]);
        let plan = plan(&view, 1.0);
        assert_eq!((plan.visible.2, plan.visible.3), (901, 701));
        assert!(holds((0, 0, SOURCE.0, SOURCE.1), plan.visible));
        assert!(plan.paint.contains_rect(tab()));
        // The painted pixels are one screen pixel each.
        assert!((plan.paint.width() - 901.0).abs() < 1e-3);
        // At 400 percent a quarter of that is seen and it is magnified.
        let deep = self::plan(&at(4.0, [0.4, 0.6]), 1.0);
        assert_eq!((deep.visible.2, deep.visible.3), (226, 176));
        assert!((deep.paint.width() - 226.0 * 4.0).abs() < 1e-2);
        assert!(deep.paint.contains_rect(tab()));
    }

    #[test]
    fn a_pan_never_resizes_what_is_seen() {
        let size = |view: &View| {
            let plan = plan(view, 1.0);
            (plan.visible.2, plan.visible.3)
        };
        let first = size(&at(1.0, [0.4, 0.6]));
        for centre in [[0.40001, 0.6], [0.4173, 0.5519], [0.0, 0.0], [1.0, 1.0]] {
            assert_eq!(size(&at(1.0, centre)), first, "{centre:?}");
        }
    }

    #[test]
    fn the_padded_window_holds_what_is_seen_with_half_a_tab_around_it() {
        let plan = plan(&at(1.0, [0.5, 0.5]), 1.0);
        assert_eq!(plan.pad, (450, 350));
        assert_eq!(plan.grid, WINDOW_GRID);
        let window = plan.window(None);
        assert!(holds(window, plan.visible));
        let (x, y, w, h) = plan.visible;
        assert!(window.0 + 450 <= x && window.1 + 350 <= y);
        assert!(window.0 + window.2 >= x + w + 450 && window.1 + window.3 >= y + h + 350);
        // Under 100 percent the pad and the grid shrink with the render.
        let half = self::plan(&at(0.5, [0.5, 0.5]), 1.0);
        assert_eq!((half.pad, half.grid), ((450, 350), 32));
    }

    #[test]
    fn the_padded_window_is_snapped_to_the_grid() {
        let window = padded_window((6000, 4000), (2551, 1651, 901, 701), (450, 350), 64);
        assert_eq!(window, (2048, 1280, 1856, 1472));
        for edge in [window.0, window.1, window.0 + window.2, window.1 + window.3] {
            assert_eq!(edge % 64, 0);
        }
    }

    #[test]
    fn the_padded_window_never_leaves_the_photo() {
        let corner = padded_window((6000, 4000), (5099, 3299, 901, 701), (450, 350), 64);
        assert_eq!(corner, (4608, 2944, 1392, 1056));
        assert!(holds((0, 0, 6000, 4000), corner));
        let origin = padded_window((6000, 4000), (0, 0, 901, 701), (450, 350), 64);
        assert_eq!((origin.0, origin.1), (0, 0));
        // A picture smaller than the pad is its own window.
        assert_eq!(
            padded_window((300, 200), (0, 0, 300, 200), (450, 350), 64),
            (0, 0, 300, 200)
        );
    }

    #[test]
    fn a_pan_inside_the_window_reuses_it_and_leaving_it_replaces_it() {
        let view = at(1.0, [0.5, 0.5]);
        let first = plan(&view, 1.0);
        let window = first.window(None);
        let rendered = Some((first.full, window));
        // 300 points is inside the pad of 450.
        let near = view.panned(tab(), SOURCE, 1.0, Vec2::new(-300.0, 200.0));
        let near = plan(&near, 1.0);
        assert_ne!(near.visible, first.visible);
        assert_eq!(near.window(rendered), window);
        // 700 points is past it: a new window around the new place.
        let far = view.panned(tab(), SOURCE, 1.0, Vec2::new(-700.0, 0.0));
        let far = plan(&far, 1.0);
        let replaced = far.window(rendered);
        assert_ne!(replaced, window);
        assert!(holds(replaced, far.visible));
    }

    #[test]
    fn a_change_of_zoom_replaces_the_window_until_the_source_is_reached() {
        let first = plan(&at(0.5, [0.5, 0.5]), 1.0);
        let rendered = Some((first.full, first.window(None)));
        let closer = plan(&at(0.625, [0.5, 0.5]), 1.0);
        assert_ne!(closer.full, first.full);
        assert_eq!(closer.window(rendered), closer.window(None));
        // From 100 percent up the render is the source either way: zooming
        // further in sees less of the same window and renders nothing new.
        let actual = plan(&at(1.0, [0.5, 0.5]), 1.0);
        let rendered = Some((actual.full, actual.window(None)));
        let deep = plan(&at(4.0, [0.5, 0.5]), 1.0);
        assert_eq!(deep.window(rendered), actual.window(None));
        // Coming back out sees more than a window made at 400 percent holds.
        let small = Some((deep.full, deep.window(None)));
        assert_ne!(actual.window(small), deep.window(None));
    }

    #[test]
    fn a_plain_drag_moves_what_it_starts_on_and_nothing_on_the_bare_picture() {
        assert_eq!(
            gesture(Over::Handle, true, false, false),
            Gesture::HandleDrag
        );
        assert_eq!(gesture(Over::Crop, true, false, false), Gesture::CropDrag);
        assert_eq!(gesture(Over::Picture, true, false, false), Gesture::Nothing);
        for over in [Over::Handle, Over::Crop, Over::Picture] {
            assert_eq!(gesture(over, false, false, false), Gesture::Nothing);
            assert_eq!(
                gesture(over, false, false, true),
                Gesture::Nothing,
                "Space alone"
            );
        }
    }

    #[test]
    fn an_armed_brush_paints_on_the_picture_on_the_crop_and_over_a_handle() {
        for under in [Over::Picture, Over::Crop, Over::Handle] {
            let armed = over(true, under);
            assert_eq!(gesture(armed, true, false, false), Gesture::Paint);
            // The pans go by, and nothing happens with no button down.
            assert_eq!(gesture(armed, true, false, true), Gesture::Pan);
            assert_eq!(gesture(armed, false, true, false), Gesture::Pan);
            assert_eq!(gesture(armed, false, false, false), Gesture::Nothing);
            assert_eq!(gesture(armed, false, false, true), Gesture::Nothing);
            // With no brush armed the drag is what it was.
            assert_eq!(over(false, under), under);
        }
        assert_eq!(
            gesture(over(false, Over::Crop), true, false, false),
            Gesture::CropDrag
        );
    }

    #[test]
    fn space_with_a_drag_and_a_middle_drag_pan_wherever_they_start() {
        for over in [Over::Handle, Over::Crop, Over::Picture, Over::Brush] {
            assert_eq!(gesture(over, true, false, true), Gesture::Pan);
            assert_eq!(gesture(over, false, true, false), Gesture::Pan);
            assert_eq!(gesture(over, false, true, true), Gesture::Pan);
        }
    }

    #[test]
    fn the_view_keys_are_ctrl_0_ctrl_1_ctrl_plus_ctrl_minus_and_f() {
        let ctrl = ViewKeys {
            command: true,
            ..ViewKeys::default()
        };
        assert_eq!(
            view_key(ViewKeys { zero: true, ..ctrl }),
            Some(ViewKey::Fit)
        );
        assert_eq!(
            view_key(ViewKeys { one: true, ..ctrl }),
            Some(ViewKey::Actual)
        );
        assert_eq!(view_key(ViewKeys { plus: true, ..ctrl }), Some(ViewKey::In));
        assert_eq!(
            view_key(ViewKeys {
                minus: true,
                ..ctrl
            }),
            Some(ViewKey::Out)
        );
        let bare = ViewKeys {
            bare: true,
            ..ViewKeys::default()
        };
        assert_eq!(view_key(ViewKeys { f: true, ..bare }), Some(ViewKey::Fit));
        // A bare digit is not a view key, and Ctrl+F is not Fit.
        assert_eq!(view_key(ViewKeys { zero: true, ..bare }), None);
        assert_eq!(view_key(ViewKeys { one: true, ..bare }), None);
        assert_eq!(view_key(ViewKeys { f: true, ..ctrl }), None);
        assert_eq!(view_key(ctrl), None);
    }

    #[test]
    fn the_view_keys_are_ignored_while_a_text_field_has_the_keyboard() {
        for keys in [
            ViewKeys {
                command: true,
                zero: true,
                ..ViewKeys::default()
            },
            ViewKeys {
                command: true,
                minus: true,
                ..ViewKeys::default()
            },
            ViewKeys {
                bare: true,
                f: true,
                ..ViewKeys::default()
            },
        ] {
            assert!(view_key(keys).is_some());
            assert_eq!(
                view_key(ViewKeys {
                    typing: true,
                    ..keys
                }),
                None
            );
        }
    }

    #[test]
    fn a_notch_of_the_wheel_is_one_step_of_a_quarter() {
        use egui::MouseWheelUnit::{Line, Point};
        assert_eq!(wheel_steps(Line, Vec2::new(0.0, 1.0)), 1.0);
        assert_eq!(wheel_steps(Line, Vec2::new(0.0, -2.0)), -2.0);
        assert_eq!(wheel_steps(Point, Vec2::new(0.0, 20.0)), 0.5);
        let pointer = Pos2::new(500.0, 400.0);
        let view = at(1.0, [0.5, 0.5]);
        let steps = wheel_steps(Line, Vec2::new(0.0, 1.0));
        let closer = view.stepped(tab(), SOURCE, 1.0, pointer, steps);
        assert_eq!(closer.zoom, Zoom::Scale(1.25));
        let back = closer.stepped(tab(), SOURCE, 1.0, pointer, -steps);
        assert!(matches!(back.zoom, Zoom::Scale(s) if (s - 1.0).abs() < 1e-6));
    }

    #[test]
    fn space_released_after_a_pan_toggles_nothing_and_after_no_pan_toggles_playback() {
        assert!(space_release_toggles(false, false));
        assert!(!space_release_toggles(true, false), "it panned");
        assert!(!space_release_toggles(false, true), "it was typed");
        assert!(!space_release_toggles(true, true));
    }

    #[test]
    fn the_keys_and_the_buttons_change_the_view_about_the_middle_of_the_tab() {
        let view = at(2.0, [0.3, 0.7]);
        assert_eq!(
            view.after_key(ViewKey::Fit, tab(), SOURCE, 1.0),
            View::default()
        );
        let actual = view.after_key(ViewKey::Actual, tab(), SOURCE, 1.0);
        assert_eq!(actual.zoom, Zoom::Scale(1.0));
        // The point in the middle of the tab stays there.
        assert!(close(actual.centre, [0.3, 0.7]));
        let closer = view.after_key(ViewKey::In, tab(), SOURCE, 1.0);
        assert_eq!(closer.zoom, Zoom::Scale(2.5));
        assert!(close(closer.centre, [0.3, 0.7]));
        let further = view.after_key(ViewKey::Out, tab(), SOURCE, 1.0);
        assert!(matches!(further.zoom, Zoom::Scale(s) if (s - 1.6).abs() < 1e-5));
        // From Fit, 100 percent lands on the middle of the picture.
        let from_fit = View::default().after_key(ViewKey::Actual, tab(), SOURCE, 1.0);
        assert_eq!(from_fit, at(1.0, [0.5, 0.5]));
    }
}
