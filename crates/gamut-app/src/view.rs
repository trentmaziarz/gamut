//! The view of the open picture: how far it is zoomed and which part of it
//! sits in the middle of the Viewer tab. A view belongs to the open file and
//! to the window; it is never saved.
//!
//! A scale is output pixels per source pixel, so 1.0 is 100 percent. The
//! least scale is the one that fits the whole picture into the tab and the
//! view at that scale is [`Zoom::Fit`], which keeps fitting when the tab is
//! resized. Everything here is arithmetic on rectangles: the viewer asks
//! where the picture goes ([`View::place`]) and the pointer code asks for a
//! view that keeps a point still ([`View::zoomed_about`]).

use egui::{Pos2, Rect, Vec2};

use crate::viewer::fit_aspect;

/// The most a picture is magnified: 800 percent, unless fitting the tab
/// already takes more.
pub const MAX_SCALE: f32 = 8.0;

/// One notch of the wheel, and one press of the zoom keys.
pub const ZOOM_STEP: f32 = 1.25;

/// A scale this close to the fit scale is the fit scale.
const FIT_SNAP: f32 = 1e-3;

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
    /// viewer placed the picture before it could zoom.
    pub fn place(&self, tab: Rect, source: (u32, u32), pixels_per_point: f32) -> Placement {
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
        let placed = self.place(tab, source, pixels_per_point);
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
        let placed = self.place(tab, source, pixels_per_point);
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
        let placed = self.place(tab, source, pixels_per_point);
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
        let image = view.place(tab(), SOURCE, pixels_per_point).image;
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
        let before = view.place(tab(), SOURCE, 1.0).image;
        let panned = view.panned(tab(), SOURCE, 1.0, Vec2::new(120.0, -80.0));
        let after = panned.place(tab(), SOURCE, 1.0).image;
        assert!((after.min - before.min - Vec2::new(120.0, -80.0)).length() < 1e-2);
        let far = view.panned(tab(), SOURCE, 1.0, Vec2::new(1e6, 1e6));
        let image = far.place(tab(), SOURCE, 1.0).image;
        assert!((image.min - tab().min).length() < 1e-2);
        // A fitted picture has nowhere to go.
        assert_eq!(
            View::default().panned(tab(), SOURCE, 1.0, Vec2::new(50.0, 50.0)),
            View::default()
        );
    }
}
