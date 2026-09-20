//! The Viewer tab: the developed photo, or the frame under the playhead of
//! the open project, from the shared device, with the crop rectangle, the
//! phone frame and its guides drawn over it. With nothing open it shows
//! the M0 test image. A video frame goes through the same develop graph
//! as a photo, so the Basic sliders and the crop apply to it.

use egui::load::SizedTexture;
use egui::{Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};
use egui_wgpu::RenderState;
use egui_wgpu::wgpu;
use gamut_core::{CropAspect, CropRect, ExportPreset, PhotoEdit};
use gamut_gpu::{Develop, TestImage};
use gamut_media::Photo;

use crate::app::Session;
use crate::mask_handles::{self, PictureMap};

/// The viewer keeps the 4:5 feed-post shape when no photo is open.
pub const VIEWER_ASPECT: [f32; 2] = [4.0, 5.0];

/// The size the viewer texture starts at, before the tab reports its size.
const INITIAL_VIEWER_SIZE: (u32, u32) = (1080, 1350);

/// The Meta safe zone of a 9:16 frame: top, bottom and each side, as
/// fractions of the frame.
pub const SAFE_ZONE: (f32, f32, f32) = (0.14, 0.35, 0.06);

/// What the egui texture currently shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shown {
    Placeholder((u32, u32)),
    /// The developed photo at a pixel size, from a given output texture.
    Photo((u32, u32), u64),
}

/// The offscreen picture and the egui texture that shows it.
pub struct Viewer {
    render_state: RenderState,
    placeholder: TestImage,
    develop: Develop,
    texture_id: egui::TextureId,
    shown: Shown,
    /// The player frame the develop graph holds.
    frame_serial: u64,
}

impl Viewer {
    pub fn new(render_state: RenderState) -> Self {
        let (width, height) = INITIAL_VIEWER_SIZE;
        let placeholder = TestImage::new(&render_state.device, width, height);
        placeholder.render(&render_state.device, &render_state.queue);
        let develop = Develop::new(&render_state.device, &render_state.queue);
        let texture_id = render_state.renderer.write().register_native_texture(
            &render_state.device,
            placeholder.view(),
            wgpu::FilterMode::Linear,
        );
        Self {
            render_state,
            placeholder,
            develop,
            texture_id,
            shown: Shown::Placeholder((width, height)),
            frame_serial: 0,
        }
    }

    /// Uploads a photo to the develop graph.
    pub fn set_photo(&mut self, photo: &Photo) {
        self.develop.set_source(photo);
        self.frame_serial = 0;
    }

    /// A project was opened: the next player frame is new.
    pub fn clear_video(&mut self) {
        self.frame_serial = 0;
    }

    /// Renders the crop for a preset at full resolution and returns the
    /// sRGB bytes, on the same device the viewer draws with.
    pub fn export(
        &mut self,
        edit: &PhotoEdit,
        crop: CropRect,
        preset: ExportPreset,
    ) -> Option<Vec<u8>> {
        self.develop.render_export(edit, crop, preset)
    }

    /// Draws the picture where the view of the open file puts it (fitted
    /// to the tab until it is zoomed), then the crop and the frame over it.
    /// The develop graph runs only when the session is dirty or the pixel
    /// size changed.
    pub fn ui(&mut self, ui: &mut egui::Ui, session: &mut Session) {
        let scale = ui.pixels_per_point();
        let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        // With nothing open the test image is fitted and never zooms.
        let placed = match session.source_size() {
            Some(source) => {
                let placed = session.view.place(area, source, scale);
                session.view = placed.view;
                Some(placed)
            }
            None => None,
        };
        let points = fit_aspect(
            area.size(),
            match session.source_size() {
                Some((width, height)) => [width as f32, height as f32],
                None => VIEWER_ASPECT,
            },
        );
        let wanted = (
            (points.x * scale).round().max(1.0) as u32,
            (points.y * scale).round().max(1.0) as u32,
        );
        // The overlay of the selected mask is part of what is shown.
        let overlay = session.adjust.overlay();
        if self.develop.overlay() != overlay {
            self.develop.set_overlay(overlay);
            session.develop_dirty = true;
        }
        let mut has_picture = false;
        if session.photo.is_some() {
            self.show_photo(wanted, session);
            has_picture = true;
        } else if session.project.is_some() {
            has_picture = self.show_video(wanted, session);
            if !has_picture {
                self.show_placeholder(wanted);
            }
        } else {
            self.show_placeholder(wanted);
        }

        let image = placed.map_or_else(
            || Rect::from_center_size(area.center(), points),
            |placed| placed.image,
        );
        egui::Image::from_texture(SizedTexture::new(self.texture_id, image.size()))
            .paint_at(ui, image);
        if let Some(placed) = placed.filter(|_| has_picture) {
            // The one place that knows where the photo is on the screen.
            let map = PictureMap::new(&placed, area);
            draw_crop(ui, &map, session);
            // The handles of the selected mask go over the crop and take the
            // pointer first.
            mask_handles::show(ui, &map, session);
        }
    }

    /// Uploads the player's current frame when it changed and draws it.
    /// `false` when no frame has arrived yet.
    fn show_video(&mut self, wanted: (u32, u32), session: &mut Session) -> bool {
        let Some(project) = session.project.as_mut() else {
            return false;
        };
        let playing = project.player.is_playing();
        if let Some((serial, frame)) = project.player.current_frame()
            && serial != self.frame_serial
        {
            self.develop
                .set_video_frame(&frame.frame, frame.colour, frame.rotation);
            self.frame_serial = serial;
            session.develop_dirty = true;
        }
        if playing {
            session.repaint_wanted = true;
        }
        if !self.develop.has_video() {
            return false;
        }
        self.show_photo(wanted, session);
        true
    }

    fn show_photo(&mut self, wanted: (u32, u32), session: &mut Session) {
        let current = Shown::Photo(wanted, self.develop.output_generation());
        if self.shown == current && !session.develop_dirty {
            return;
        }
        let view = self
            .develop
            .render(&session.edit, CropRect::FULL, wanted, wanted)
            .expect("a photo is set")
            .clone();
        let now = Shown::Photo(wanted, self.develop.output_generation());
        if self.shown != now {
            self.render_state
                .renderer
                .write()
                .update_egui_texture_from_wgpu_texture(
                    &self.render_state.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    self.texture_id,
                );
            self.shown = now;
        }
        session.develop_dirty = false;
    }

    fn show_placeholder(&mut self, wanted: (u32, u32)) {
        if self.shown == Shown::Placeholder(wanted) {
            return;
        }
        let device = &self.render_state.device;
        self.placeholder.resize(device, wanted.0, wanted.1);
        self.placeholder.render(device, &self.render_state.queue);
        self.render_state
            .renderer
            .write()
            .update_egui_texture_from_wgpu_texture(
                device,
                self.placeholder.view(),
                wgpu::FilterMode::Linear,
                self.texture_id,
            );
        self.shown = Shown::Placeholder(wanted);
    }
}

/// The crop after a drag of `delta` points across the screen: it moves by
/// that distance on the photo, whatever the zoom.
pub fn dragged_crop(map: &PictureMap, crop: CropRect, delta: Vec2) -> CropRect {
    let [dx, dy] = map.delta_to_picture(delta);
    crop.moved(dx, dy)
}

/// The crop rectangle, draggable, under the phone frame and its guides.
fn draw_crop(ui: &mut egui::Ui, map: &PictureMap, session: &mut Session) {
    let id = ui.id().with("crop drag");
    // Only the part of the crop inside the tab takes the pointer; a drag
    // under way keeps going when the pointer leaves it.
    let grab = map
        .touchable(map.rect_to_screen(session.crop.rect))
        .or_else(|| ui.ctx().is_being_dragged(id).then_some(map.visible));
    if let Some(grab) = grab {
        let response = ui.interact(grab, id, Sense::drag());
        if response.dragged() {
            session.crop.rect = dragged_crop(map, session.crop.rect, response.drag_delta());
            session.mark_edited();
        }
    }
    let image = map.image;
    let crop = map.rect_to_screen(session.crop.rect);
    let painter = ui.painter().with_clip_rect(map.clip(12.0));

    // Everything outside the crop is dimmed.
    let shade = Color32::from_black_alpha(130);
    let bands = [
        Rect::from_min_max(image.min, Pos2::new(image.max.x, crop.min.y)),
        Rect::from_min_max(Pos2::new(image.min.x, crop.max.y), image.max),
        Rect::from_min_max(
            Pos2::new(image.min.x, crop.min.y),
            Pos2::new(crop.min.x, crop.max.y),
        ),
        Rect::from_min_max(
            Pos2::new(crop.max.x, crop.min.y),
            Pos2::new(image.max.x, crop.max.y),
        ),
    ];
    for band in bands {
        if band.is_positive() {
            painter.rect_filled(band, 0, shade);
        }
    }

    // The phone frame around the crop.
    painter.rect_stroke(
        crop.expand(5.0),
        CornerRadius::same(14),
        Stroke::new(8.0, Color32::from_gray(24)),
        StrokeKind::Outside,
    );
    painter.rect_stroke(
        crop,
        0,
        Stroke::new(1.5, Color32::WHITE),
        StrokeKind::Inside,
    );

    match session.crop.aspect {
        CropAspect::Story9x16 => draw_safe_zone(&painter, crop),
        CropAspect::Feed4x5 if session.grid_guide => draw_grid_guide(&painter, crop),
        _ => {}
    }
}

/// The Meta safe zone on a 9:16 frame: the bands the UI covers are
/// tinted and the zone inside them is outlined.
fn draw_safe_zone(painter: &egui::Painter, crop: Rect) {
    let (top, bottom, side) = SAFE_ZONE;
    let safe = Rect::from_min_max(
        Pos2::new(
            crop.min.x + side * crop.width(),
            crop.min.y + top * crop.height(),
        ),
        Pos2::new(
            crop.max.x - side * crop.width(),
            crop.max.y - bottom * crop.height(),
        ),
    );
    let tint = Color32::from_rgba_unmultiplied(255, 140, 0, 70);
    let bands = [
        Rect::from_min_max(crop.min, Pos2::new(crop.max.x, safe.min.y)),
        Rect::from_min_max(Pos2::new(crop.min.x, safe.max.y), crop.max),
        Rect::from_min_max(
            Pos2::new(crop.min.x, safe.min.y),
            Pos2::new(safe.min.x, safe.max.y),
        ),
        Rect::from_min_max(
            Pos2::new(safe.max.x, safe.min.y),
            Pos2::new(crop.max.x, safe.max.y),
        ),
    ];
    for band in bands {
        painter.rect_filled(band, 0, tint);
    }
    painter.rect_stroke(
        safe,
        0,
        Stroke::new(1.0, Color32::from_rgb(255, 170, 40)),
        StrokeKind::Inside,
    );
}

/// The centred 3:4 crop the profile grid shows of a 4:5 post.
fn draw_grid_guide(painter: &egui::Painter, crop: Rect) {
    let grid_width = crop.height() * 0.75;
    let inset = (crop.width() - grid_width) / 2.0;
    let stroke = Stroke::new(1.0, Color32::from_rgb(80, 200, 255));
    for x in [crop.min.x + inset, crop.max.x - inset] {
        painter.line_segment([Pos2::new(x, crop.min.y), Pos2::new(x, crop.max.y)], stroke);
    }
}

/// The largest size of the given aspect that fits inside `available`.
pub fn fit_aspect(available: egui::Vec2, aspect: [f32; 2]) -> egui::Vec2 {
    let ratio = aspect[0] / aspect[1];
    let width = available.x.min(available.y * ratio).max(0.0);
    egui::vec2(width, width / ratio)
}

#[cfg(test)]
mod tests {
    use super::{VIEWER_ASPECT, dragged_crop, fit_aspect};
    use crate::mask_handles::PictureMap;
    use crate::view::{View, Zoom};
    use egui::{Pos2, Rect, Vec2};
    use gamut_core::CropRect;

    #[test]
    fn a_wide_tab_is_limited_by_its_height() {
        let size = fit_aspect(egui::vec2(1000.0, 500.0), VIEWER_ASPECT);
        assert_eq!(size, egui::vec2(400.0, 500.0));
    }

    #[test]
    fn a_tall_tab_is_limited_by_its_width() {
        let size = fit_aspect(egui::vec2(400.0, 2000.0), VIEWER_ASPECT);
        assert_eq!(size, egui::vec2(400.0, 500.0));
    }

    #[test]
    fn the_crop_maps_onto_the_image_rectangle() {
        let image = Rect::from_min_max(Pos2::new(100.0, 50.0), Pos2::new(300.0, 250.0));
        let crop = CropRect {
            x: 0.25,
            y: 0.0,
            width: 0.5,
            height: 1.0,
        };
        let on_screen = PictureMap::whole(image).rect_to_screen(crop);
        assert_eq!(on_screen.min, Pos2::new(150.0, 50.0));
        assert_eq!(on_screen.max, Pos2::new(250.0, 250.0));
    }

    #[test]
    fn a_crop_drag_moves_the_crop_by_the_dragged_distance_on_the_photo() {
        let tab = Rect::from_min_size(Pos2::new(40.0, 30.0), Vec2::new(900.0, 700.0));
        let crop = CropRect {
            x: 0.2,
            y: 0.3,
            width: 0.3,
            height: 0.4,
        };
        for scale in [1.0, 4.0] {
            let view = View {
                zoom: Zoom::Scale(scale),
                centre: [0.31, 0.64],
            };
            let map = PictureMap::new(&view.place(tab, (6000, 4000), 1.0), tab);
            let before = map.rect_to_screen(crop);
            let drag = Vec2::new(60.0, -24.0);
            let moved = dragged_crop(&map, crop, drag);
            // On the screen the crop went where the pointer went.
            let after = map.rect_to_screen(moved);
            assert!((after.min - before.min - drag).length() < 1e-2);
            // On the photo that is fewer source pixels the deeper the zoom.
            assert!((moved.x - crop.x - 60.0 / (6000.0 * scale)).abs() < 1e-6);
            assert!((moved.y - crop.y + 24.0 / (4000.0 * scale)).abs() < 1e-6);
            assert_eq!((moved.width, moved.height), (crop.width, crop.height));
        }
    }
}
