//! The Viewer tab: the developed photo from the shared device, with the
//! crop rectangle, the phone frame and its guides drawn over it. With no
//! photo open it shows the M0 test image.

use egui::load::SizedTexture;
use egui::{Color32, CornerRadius, Pos2, Rect, Sense, Stroke, StrokeKind};
use egui_wgpu::RenderState;
use egui_wgpu::wgpu;
use slate_core::{CropAspect, CropRect, ExportPreset, PhotoEdit};
use slate_gpu::{Develop, TestImage};
use slate_media::Photo;

use crate::app::Session;

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
        }
    }

    /// Uploads a photo to the develop graph.
    pub fn set_photo(&mut self, photo: &Photo) {
        self.develop.set_source(photo);
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

    /// Draws the picture at the largest size of its aspect that fits the
    /// tab, then the crop and the frame over it. The develop graph runs
    /// only when the session is dirty or the pixel size changed.
    pub fn ui(&mut self, ui: &mut egui::Ui, session: &mut Session) {
        let aspect = match &session.photo {
            Some(photo) => [photo.width as f32, photo.height as f32],
            None => VIEWER_ASPECT,
        };
        let points = fit_aspect(ui.available_size(), aspect);
        let scale = ui.pixels_per_point();
        let wanted = (
            (points.x * scale).round().max(1.0) as u32,
            (points.y * scale).round().max(1.0) as u32,
        );
        if session.photo.is_some() {
            self.show_photo(wanted, session);
        } else {
            self.show_placeholder(wanted);
        }

        let (area, _) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
        let image = Rect::from_center_size(area.center(), points);
        egui::Image::from_texture(SizedTexture::new(self.texture_id, points)).paint_at(ui, image);
        if session.photo.is_some() {
            draw_crop(ui, image, session);
        }
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

/// The crop rectangle in screen points, given the image rectangle.
fn crop_on_screen(image: Rect, crop: CropRect) -> Rect {
    Rect::from_min_size(
        Pos2::new(
            image.min.x + crop.x * image.width(),
            image.min.y + crop.y * image.height(),
        ),
        egui::vec2(crop.width * image.width(), crop.height * image.height()),
    )
}

/// The crop rectangle, draggable, under the phone frame and its guides.
fn draw_crop(ui: &mut egui::Ui, image: Rect, session: &mut Session) {
    let crop = crop_on_screen(image, session.crop.rect);
    let response = ui.interact(crop, ui.id().with("crop drag"), Sense::drag());
    if response.dragged() {
        let delta = response.drag_delta();
        session.crop.rect = session
            .crop
            .rect
            .moved(delta.x / image.width(), delta.y / image.height());
        session.mark_edited();
    }
    let crop = crop_on_screen(image, session.crop.rect);
    let painter = ui.painter().with_clip_rect(image.expand(12.0));

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
    use super::{VIEWER_ASPECT, crop_on_screen, fit_aspect};
    use egui::{Pos2, Rect};
    use slate_core::CropRect;

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
        let on_screen = crop_on_screen(image, crop);
        assert_eq!(on_screen.min, Pos2::new(150.0, 50.0));
        assert_eq!(on_screen.max, Pos2::new(250.0, 250.0));
    }
}
