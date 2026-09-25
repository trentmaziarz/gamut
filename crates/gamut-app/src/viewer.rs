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
use gamut_gpu::{Develop, TestImage, ViewWindow};
use gamut_media::Photo;

use crate::app::Session;
use crate::brush_tool;
use crate::mask_handles::{self, PictureMap};
use crate::view::{self, Gesture, Over, PanInput, RenderPlan, ViewKeys, Zoom};

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
    /// The developed photo at a pixel size, from a given output texture,
    /// and whether the texture is registered with the nearest filter.
    Photo((u32, u32), u64, bool),
}

/// What the develop graph is asked to render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    /// The whole picture at the fitted size: a Fit view.
    Whole((u32, u32)),
    /// A padded window of a zoomed view and what is seen of it.
    Window(ViewWindow),
}

impl Target {
    fn output_size(&self) -> (u32, u32) {
        match self {
            Target::Whole(size) => *size,
            Target::Window(view) => (view.visible.2, view.visible.3),
        }
    }
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
    /// What the develop graph rendered last: a pan changes it without
    /// touching the session.
    rendered: Option<Target>,
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
            rendered: None,
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
        // The wheel, the keys and a pan on the bare picture move the view
        // before anything is placed, so the picture follows in this frame.
        let mut pan = PanInput::default();
        if let Some(source) = session.source_size() {
            view_input(ui, area, source, session, &mut pan);
        }
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
        // A zoomed view renders a window of the source; a Fit view renders
        // the whole picture at the fitted size, exactly as before.
        let plan =
            placed.and_then(|placed| RenderPlan::of(&placed, area, session.source_size()?, scale));
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
        // Once what is seen leaves the rendered window, the window built
        // ahead of the pan is taken when it holds what is seen.
        let target = match plan {
            Some(plan) => Target::Window(ViewWindow {
                full: plan.full,
                window: plan.window_or_ahead(self.window_rendered(), self.develop.window_ahead()),
                visible: plan.visible,
            }),
            None => Target::Whole(wanted),
        };
        // What the last render showed, for the direction of a pan.
        let previous = self.rendered;
        let nearest = plan.is_some_and(|plan| plan.nearest);
        let mut has_picture = false;
        if session.photo.is_some() {
            self.show_photo(target, nearest, session);
            has_picture = true;
        } else if session.project.is_some() {
            has_picture = self.show_video(target, nearest, session);
            if !has_picture {
                self.show_placeholder(wanted);
            }
        } else {
            self.show_placeholder(wanted);
        }
        if has_picture && let Some(plan) = plan {
            self.build_ahead(&plan, previous, session);
        }

        // Zoomed, the texture holds only what is seen and goes where that
        // is; otherwise it holds the whole picture, or the test image.
        let paint = match (placed, plan) {
            (Some(_), Some(plan)) if has_picture => plan.paint,
            (Some(placed), None) if has_picture => placed.image,
            _ => Rect::from_center_size(area.center(), points),
        };
        egui::Image::from_texture(SizedTexture::new(self.texture_id, paint.size()))
            .paint_at(ui, paint);
        if let Some(placed) = placed.filter(|_| has_picture) {
            // The one place that knows where the photo is on the screen.
            let map = PictureMap::new(&placed, area);
            picture_input(ui, &map, session, &mut pan);
        }
        // A pan that began on the crop or on a handle is known only now.
        if let Some(source) = session.source_size() {
            apply_pan(ui, area, source, session, &mut pan);
            if pan.panning {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
            } else if pan.space && ui.rect_contains_pointer(area) {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
        }
        session.view_link.shown = placed
            .filter(|_| has_picture)
            .map(|placed| (placed.scale, placed.view.zoom == Zoom::Fit));
    }

    /// Uploads the player's current frame when it changed and draws it.
    /// `false` when no frame has arrived yet.
    fn show_video(&mut self, target: Target, nearest: bool, session: &mut Session) -> bool {
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
        self.show_photo(target, nearest, session);
        true
    }

    /// Builds the window a pan will reach ahead of it, from the step between
    /// what the render before showed and what this one shows, when the view
    /// moved inside the same picture size. A window already built or
    /// building for where the pan leaves is kept; the graph builds it again
    /// only when the edit moved the frame it needs. The graph records one
    /// slice of that frame a call, so this runs once a UI frame while it is
    /// building, through a pause in the pan too, and asks for the next UI
    /// frame until the last slice is submitted.
    fn build_ahead(&mut self, plan: &RenderPlan, previous: Option<Target>, session: &mut Session) {
        let (Some(Target::Window(before)), Some(Target::Window(now))) = (previous, self.rendered)
        else {
            return;
        };
        // A step of more than the pad is a jump, not a pan.
        if before.full != now.full
            || before.visible.0.abs_diff(now.visible.0) > plan.pad.0
            || before.visible.1.abs_diff(now.visible.1) > plan.pad.1
        {
            return;
        }
        let built = self
            .develop
            .window_ahead()
            .filter(|(full, _)| *full == now.full)
            .map(|(_, window)| window);
        let wanted = if before.visible == now.visible {
            // The pan paused: a frame ahead still building goes on.
            built.filter(|_| self.develop.ahead_building())
        } else {
            plan.ahead(now.window, before.visible, built)
        };
        if let Some(ahead) = wanted {
            self.develop.build_ahead(&session.edit, now.full, ahead);
            if self.develop.ahead_building() {
                session.repaint_wanted = true;
            }
        }
    }

    /// The window the develop graph holds, for a zoomed view to keep.
    fn window_rendered(&self) -> Option<((u32, u32), crate::view::PixelRect)> {
        match self.rendered {
            Some(Target::Window(view)) => Some((view.full, view.window)),
            _ => None,
        }
    }

    fn show_photo(&mut self, target: Target, nearest: bool, session: &mut Session) {
        let size = target.output_size();
        let current = Shown::Photo(size, self.develop.output_generation(), nearest);
        if self.shown == current && self.rendered == Some(target) && !session.develop_dirty {
            return;
        }
        let view = match &target {
            Target::Whole(wanted) => {
                self.develop
                    .render(&session.edit, CropRect::FULL, *wanted, *wanted)
            }
            Target::Window(window) => self.develop.render_view(&session.edit, window),
        }
        .expect("a photo is set")
        .clone();
        self.rendered = Some(target);
        let now = Shown::Photo(size, self.develop.output_generation(), nearest);
        if self.shown != now {
            // Magnified far enough, single pixels are judged: no filtering.
            let filter = if nearest {
                wgpu::FilterMode::Nearest
            } else {
                wgpu::FilterMode::Linear
            };
            self.render_state
                .renderer
                .write()
                .update_egui_texture_from_wgpu_texture(
                    &self.render_state.device,
                    &view,
                    filter,
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

/// Moves the view by what the pans of this frame collected, and notes a
/// Space that panned so its release does not play or pause.
fn apply_pan(
    ui: &egui::Ui,
    area: Rect,
    source: (u32, u32),
    session: &mut Session,
    pan: &mut PanInput,
) {
    let delta = std::mem::take(&mut pan.delta);
    if delta == Vec2::ZERO {
        return;
    }
    log::trace!("pan by {delta:?}, space {}", pan.space);
    if pan.space {
        session.view_link.space_panned = true;
    }
    let moved = session
        .view
        .panned(area, source, ui.pixels_per_point(), delta);
    if moved != session.view {
        session.view = moved;
        ui.ctx().request_repaint();
    }
}

/// The input that changes the view: the wheel and a pinch over the picture
/// zoom about the pointer, the view keys and the buttons of the Adjust tab
/// act about the middle of the tab, and a middle drag or a drag with Space
/// held on the bare picture pans. None of it runs while the Save, Discard,
/// Cancel prompt is up.
fn view_input(
    ui: &mut egui::Ui,
    area: Rect,
    source: (u32, u32),
    session: &mut Session,
    pan: &mut PanInput,
) {
    let scale = ui.pixels_per_point();
    // A slider keeps the focus after a click, so focus alone is not typing.
    let typing = ui.ctx().text_edit_focused();
    let prompt = session.adjust.pending_switch.is_some();
    let (keys, space, space_released, steps, pinch, pointer) = ui.input(|i| {
        let mut steps = 0.0;
        let mut pinch = 1.0;
        for event in &i.events {
            match event {
                egui::Event::MouseWheel { unit, delta, .. } => {
                    steps += view::wheel_steps(*unit, *delta);
                }
                egui::Event::Zoom(factor) => pinch *= *factor,
                _ => {}
            }
        }
        let modifiers = i.modifiers;
        let keys = ViewKeys {
            command: modifiers.command && !modifiers.alt,
            bare: modifiers.is_none(),
            zero: i.key_pressed(egui::Key::Num0),
            one: i.key_pressed(egui::Key::Num1),
            plus: i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals),
            minus: i.key_pressed(egui::Key::Minus),
            f: i.key_pressed(egui::Key::F),
            typing,
        };
        (
            keys,
            i.key_down(egui::Key::Space),
            i.key_released(egui::Key::Space),
            steps,
            pinch,
            i.pointer.hover_pos(),
        )
    });
    pan.space = space && !typing && !prompt;
    // The release belongs to the timeline, which reads the note this frame;
    // a Space that is simply up has nothing left to say.
    if !space && !space_released {
        session.view_link.space_panned = false;
    }
    // The bare picture, under the crop and the handles: it pans and does
    // nothing else. An armed brush covers the same rectangle and reports the
    // pans itself, and egui gives a drag to the earlier of two drag widgets
    // of one rectangle (it takes the later for a background), so with a
    // brush in the hand this widget is not made and the brush's is the one.
    if session.adjust.brush.armed.is_none() {
        let bare = ui.interact(area, ui.id().with("view pan"), Sense::drag());
        if pan.take(&bare, Over::Picture) == Gesture::Pan {
            apply_pan(ui, area, source, session, pan);
        }
    }
    if prompt {
        session.view_link.request = None;
        return;
    }
    let mut view = session.view;
    if let Some(key) = session.view_link.request.take().or(view::view_key(keys)) {
        view = view.after_key(key, area, source, scale);
    }
    if let Some(pointer) = pointer.filter(|_| ui.rect_contains_pointer(area)) {
        if steps != 0.0 {
            log::trace!("wheel {steps} at {pointer:?}");
            view = view.stepped(area, source, scale, pointer, steps);
        }
        if pinch != 1.0 {
            let shown = view.place_exact(area, source, scale).scale;
            view = view.zoomed_about(area, source, scale, pointer, shown * pinch);
        }
    }
    if view != session.view {
        session.view = view;
        ui.ctx().request_repaint();
    }
}

/// Everything on the picture that takes the pointer, bottom to top: the
/// crop, the handles of the selected mask over it, and an armed brush over
/// everything, which takes the plain drag.
fn picture_input(ui: &mut egui::Ui, map: &PictureMap, session: &mut Session, pan: &mut PanInput) {
    draw_crop(ui, map, session, pan);
    mask_handles::show(ui, map, session, pan);
    brush_tool::show(ui, map, session, pan);
}

/// The crop rectangle, draggable, under the phone frame and its guides.
fn draw_crop(ui: &mut egui::Ui, map: &PictureMap, session: &mut Session, pan: &mut PanInput) {
    let id = ui.id().with("crop drag");
    // Only the part of the crop inside the tab takes the pointer; a drag
    // under way keeps going when the pointer leaves it. An armed brush
    // paints inside the crop too, so the crop stays where it is.
    let grab = map
        .touchable(map.rect_to_screen(session.crop.rect))
        .or_else(|| ui.ctx().is_being_dragged(id).then_some(map.visible))
        .filter(|_| view::over(session.adjust.brush.armed.is_some(), Over::Crop) == Over::Crop);
    if let Some(grab) = grab {
        let response = ui.interact(grab, id, Sense::drag());
        // A plain drag moves the crop, as it always did; the pans go by.
        if pan.take(&response, Over::Crop) == Gesture::CropDrag {
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
    use super::{VIEWER_ASPECT, dragged_crop, fit_aspect, picture_input, view_input};
    use crate::app::Session;
    use crate::mask_handles::PictureMap;
    use crate::view::{PanInput, View, Zoom};
    use egui::{Pos2, Rect, Vec2};
    use gamut_core::brush::Brush;
    use gamut_core::{CropRect, Mask, MaskSource};

    const SOURCE: (u32, u32) = (1200, 800);

    /// One frame of the picture's input as the Viewer runs it, without the
    /// render: the view input, then the crop, the handles and the brush.
    fn frame(ctx: &egui::Context, session: &mut Session, events: Vec<egui::Event>) {
        frame_with(ctx, session, events, egui::Modifiers::NONE);
    }

    /// The same with keys held: egui reads them from the frame, not the event.
    fn frame_with(
        ctx: &egui::Context,
        session: &mut Session,
        events: Vec<egui::Event>,
        modifiers: egui::Modifiers,
    ) {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 800.0))),
            // The keys held come first, as the window sends them.
            events: std::iter::once(egui::Event::ModifiersChanged(modifiers))
                .chain(events)
                .collect(),
            ..Default::default()
        };
        // No renderer takes the font texture here, so it is let go by hand.
        let mut output = ctx.run_ui(input, |ui| {
            let area = ui.max_rect();
            let mut pan = PanInput::default();
            view_input(ui, area, SOURCE, session, &mut pan);
            let placed = session.view.place(area, SOURCE, 1.0);
            session.view = placed.view;
            let map = PictureMap::new(&placed, area);
            picture_input(ui, &map, session, &mut pan);
        });
        output.textures_delta.clear();
    }

    fn button(pos: Pos2, pressed: bool, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers,
        }
    }

    /// Presses at `from`, moves to `to` over ten frames and lets go.
    fn drag(
        ctx: &egui::Context,
        session: &mut Session,
        from: Pos2,
        to: Pos2,
        modifiers: egui::Modifiers,
    ) {
        frame(ctx, session, vec![egui::Event::PointerMoved(from)]);
        frame_with(ctx, session, vec![button(from, true, modifiers)], modifiers);
        for step in 1..=10 {
            let at = from + (to - from) * (step as f32 / 10.0);
            frame_with(ctx, session, vec![egui::Event::PointerMoved(at)], modifiers);
        }
        frame_with(ctx, session, vec![button(to, false, modifiers)], modifiers);
        frame(ctx, session, Vec::new());
    }

    fn strokes(session: &Session) -> &Brush {
        match &session.edit.masks[0].components[0].source {
            MaskSource::Brush(brush) => brush,
            _ => panic!("a brush"),
        }
    }

    /// The window's own path from the pointer to a stroke. It was found
    /// broken by driving the window: the bare pan widget has the rectangle
    /// of the brush's and took its drags, so nothing was ever painted.
    #[test]
    fn a_drag_on_the_picture_paints_with_a_brush_in_the_hand_and_moves_the_crop_without() {
        let ctx = egui::Context::default();
        let mut session = Session::default();
        session
            .edit
            .masks
            .push(Mask::new("Brush", MaskSource::Brush(Brush::default())));
        session.adjust.select_mask(Some(0));
        // A crop with room to move, under where the pointer will press.
        session.crop.rect = CropRect {
            x: 0.3,
            y: 0.2,
            width: 0.4,
            height: 0.6,
        };
        let crop = session.crop.rect;
        let none = egui::Modifiers::NONE;
        // Warm up: egui tests a pointer against the widgets of the frame before.
        frame(&ctx, &mut session, Vec::new());

        // No brush in the hand: a plain drag inside the crop moves the crop.
        let (from, to) = (Pos2::new(560.0, 400.0), Pos2::new(640.0, 430.0));
        drag(&ctx, &mut session, from, to, none);
        assert_ne!(session.crop.rect, crop, "the crop moved");
        assert!(strokes(&session).strokes.is_empty());

        // The brush in the hand: the same drag paints and the crop stays.
        let crop = session.crop.rect;
        session.toggle_brush(0);
        frame(&ctx, &mut session, Vec::new());
        drag(&ctx, &mut session, from, to, none);
        assert_eq!(session.crop.rect, crop, "the crop stays where it is");
        assert_eq!(strokes(&session).strokes.len(), 1, "one drag, one stroke");
        let stroke = &strokes(&session).strokes[0];
        assert!(stroke.points.len() >= 3, "{} points", stroke.points.len());
        assert!(!stroke.erase);
        assert!(!session.adjust.brush.is_painting(), "the release ends it");
        // The stroke is where the pointer was: the fitted photo fills the
        // tab, so the press is that share of the way across it.
        let first = stroke.points[0];
        assert!((first[0] - 560.0 / 1200.0).abs() < 0.02, "{first:?}");
        assert!((first[1] - 400.0 / 800.0).abs() < 0.02, "{first:?}");

        // A click without a move is one dab, Alt erases, Shift draws a line
        // from the end of the last stroke.
        drag(
            &ctx,
            &mut session,
            Pos2::new(300.0, 300.0),
            Pos2::new(300.0, 300.0),
            none,
        );
        assert_eq!(strokes(&session).strokes[1].points.len(), 1);
        drag(&ctx, &mut session, from, to, egui::Modifiers::ALT);
        assert!(strokes(&session).strokes[2].erase);
        let end = *strokes(&session).strokes[2].points.last().expect("points");
        let at = Pos2::new(900.0, 200.0);
        drag(&ctx, &mut session, at, at, egui::Modifiers::SHIFT);
        let line = &strokes(&session).strokes[3];
        assert_eq!(line.points.len(), 2);
        assert_eq!(line.points[0], end);

        // Put down, the drag is the crop's again.
        session.put_brush_down();
        frame(&ctx, &mut session, Vec::new());
        drag(&ctx, &mut session, from, to, none);
        assert_ne!(session.crop.rect, crop);
        assert_eq!(strokes(&session).strokes.len(), 4);
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

    /// A touch from `from` to `to` over ten frames, as egui-winit sends one:
    /// the touch event with its force first, then the pointer events of the
    /// mouse it stands in for. `force` gives the force at each step, 0 to 10.
    fn touch_drag(
        ctx: &egui::Context,
        session: &mut Session,
        from: Pos2,
        to: Pos2,
        force: &dyn Fn(usize) -> Option<f32>,
    ) {
        use egui::TouchPhase::{End, Move, Start};
        let none = egui::Modifiers::NONE;
        frame(
            ctx,
            session,
            vec![
                touch(1, Start, from, force(0)),
                egui::Event::PointerMoved(from),
                button(from, true, none),
            ],
        );
        for step in 1..=10 {
            let at = from + (to - from) * (step as f32 / 10.0);
            frame(
                ctx,
                session,
                vec![
                    touch(1, Move, at, force(step)),
                    egui::Event::PointerMoved(at),
                ],
            );
        }
        frame(
            ctx,
            session,
            vec![
                touch(1, End, to, None),
                button(to, false, none),
                egui::Event::PointerGone,
            ],
        );
        frame(ctx, session, Vec::new());
    }

    fn with_a_brush_in_the_hand() -> (egui::Context, Session) {
        let ctx = egui::Context::default();
        let mut session = Session::default();
        session
            .edit
            .masks
            .push(Mask::new("Brush", MaskSource::Brush(Brush::default())));
        session.adjust.select_mask(Some(0));
        // Warm up: egui tests a pointer against the widgets of the frame before.
        frame(&ctx, &mut session, Vec::new());
        session.toggle_brush(0);
        frame(&ctx, &mut session, Vec::new());
        (ctx, session)
    }

    /// A pen reaches the brush twice in a frame, as a touch and as a pointer.
    /// The stroke takes each point once, with the force the pen had there.
    #[test]
    fn a_pen_stroke_stores_each_point_once_with_its_pressure() {
        let (ctx, mut session) = with_a_brush_in_the_hand();
        session.adjust.brush.pressure_size = true;
        let (from, to) = (Pos2::new(300.0, 400.0), Pos2::new(800.0, 450.0));
        let rising = |step: usize| Some(0.2 + 0.07 * step as f32);
        touch_drag(&ctx, &mut session, from, to, &rising);

        assert_eq!(strokes(&session).strokes.len(), 1, "one touch, one stroke");
        let stroke = &strokes(&session).strokes[0];
        assert_eq!(
            stroke.points.len(),
            11,
            "the press and ten moves, each once"
        );
        assert_eq!(stroke.pressure.len(), stroke.points.len());
        for pair in stroke.points.windows(2) {
            assert!(pair[1][0] > pair[0][0], "no point twice: {pair:?}");
        }
        assert_eq!(stroke.pressure[0], 0.2);
        assert_eq!(stroke.pressure[10], 0.9);
        assert!(stroke.pressure.windows(2).all(|pair| pair[1] > pair[0]));
        assert!(
            stroke.pressure_size && stroke.pressure_flow,
            "the toggles at the press"
        );
        assert!(!session.adjust.brush.is_painting(), "the lift ends it");
    }

    #[test]
    fn a_mouse_and_a_finger_store_no_pressure() {
        let (ctx, mut session) = with_a_brush_in_the_hand();
        let (from, to) = (Pos2::new(300.0, 400.0), Pos2::new(800.0, 450.0));
        drag(&ctx, &mut session, from, to, egui::Modifiers::NONE);
        touch_drag(&ctx, &mut session, from, to, &|_| None);
        let brush = strokes(&session);
        assert_eq!(brush.strokes.len(), 2);
        for stroke in &brush.strokes {
            assert!(stroke.points.len() >= 3);
            assert!(
                stroke.pressure.is_empty(),
                "full pressure, and nothing in the file"
            );
            assert!(!stroke.uses_pressure());
        }
        // A finger paints where a mouse does.
        assert_eq!(brush.strokes[0].points[0], brush.strokes[1].points[0]);

        // A mouse after a pen: the pen's force is not the mouse's.
        touch_drag(&ctx, &mut session, from, to, &|_| Some(0.3));
        drag(&ctx, &mut session, from, to, egui::Modifiers::NONE);
        let brush = strokes(&session);
        assert!(!brush.strokes[2].pressure.is_empty());
        assert!(brush.strokes[3].pressure.is_empty());
    }

    /// Two fingers are a pinch, which the view owns: the stroke ends where it
    /// is and takes nothing more.
    #[test]
    fn a_second_touch_ends_the_stroke_where_it_is() {
        use egui::TouchPhase::{End, Move, Start};
        let (ctx, mut session) = with_a_brush_in_the_hand();
        let none = egui::Modifiers::NONE;
        let at = |step: f32| Pos2::new(300.0 + 40.0 * step, 400.0);
        frame(
            &ctx,
            &mut session,
            vec![
                touch(1, Start, at(0.0), None),
                egui::Event::PointerMoved(at(0.0)),
                button(at(0.0), true, none),
            ],
        );
        for step in 1..=3 {
            let place = at(step as f32);
            frame(
                &ctx,
                &mut session,
                vec![
                    touch(1, Move, place, None),
                    egui::Event::PointerMoved(place),
                ],
            );
        }
        assert!(session.adjust.brush.is_painting());
        let held = strokes(&session).strokes[0].points.len();
        assert!(held >= 3);

        let other = Pos2::new(700.0, 600.0);
        frame(&ctx, &mut session, vec![touch(2, Start, other, None)]);
        assert!(
            !session.adjust.brush.is_painting(),
            "the second finger ends it"
        );
        for step in 4..=8 {
            let place = at(step as f32);
            frame(
                &ctx,
                &mut session,
                vec![
                    touch(1, Move, place, None),
                    egui::Event::PointerMoved(place),
                    touch(2, Move, other, None),
                ],
            );
        }
        frame(
            &ctx,
            &mut session,
            vec![
                touch(2, End, other, None),
                touch(1, End, at(8.0), None),
                button(at(8.0), false, none),
                egui::Event::PointerGone,
            ],
        );
        let brush = strokes(&session);
        assert_eq!(brush.strokes.len(), 1, "and starts no other");
        assert_eq!(brush.strokes[0].points.len(), held, "nothing more is taken");
    }

    fn key(key: egui::Key) -> egui::Event {
        egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }
    }

    /// A is Auto mask while a brush is in the hand: the next stroke is an
    /// auto stroke with the sensitivity of the tool, and the stroke before it
    /// stays what it was.
    #[test]
    fn the_a_key_makes_the_next_stroke_an_auto_stroke_and_leaves_the_last_alone() {
        let (ctx, mut session) = with_a_brush_in_the_hand();
        let (from, to) = (Pos2::new(300.0, 400.0), Pos2::new(800.0, 450.0));
        drag(&ctx, &mut session, from, to, egui::Modifiers::NONE);
        assert!(!strokes(&session).strokes[0].auto);

        session.adjust.brush.sensitivity = 80.0;
        frame(&ctx, &mut session, vec![key(egui::Key::A)]);
        assert!(session.adjust.brush.auto, "A ticks Auto mask");
        assert!(
            !strokes(&session).strokes[0].auto,
            "the last stroke is left alone"
        );
        drag(&ctx, &mut session, from, to, egui::Modifiers::NONE);
        let brush = strokes(&session);
        assert!(brush.strokes[1].auto);
        assert_eq!(brush.strokes[1].sensitivity, 80.0);
        assert!(!brush.strokes[0].auto);

        frame(&ctx, &mut session, vec![key(egui::Key::A)]);
        assert!(!session.adjust.brush.auto, "and A again unticks it");
        // With no brush in the hand A is nobody's.
        session.put_brush_down();
        frame(&ctx, &mut session, vec![key(egui::Key::A)]);
        assert!(!session.adjust.brush.auto);
    }

    /// With the Save, Discard, Cancel prompt up the brush is put down: a drag
    /// and a pen paint nothing.
    #[test]
    fn with_the_prompt_up_nothing_paints() {
        let (ctx, mut session) = with_a_brush_in_the_hand();
        session.adjust.brush.auto = true;
        session.adjust.pending_switch = Some("Teal".to_string());
        let (from, to) = (Pos2::new(300.0, 400.0), Pos2::new(800.0, 450.0));
        drag(&ctx, &mut session, from, to, egui::Modifiers::NONE);
        touch_drag(&ctx, &mut session, from, to, &|_| Some(0.5));
        assert!(strokes(&session).strokes.is_empty());
        assert_eq!(session.adjust.brush.armed, None, "the brush was put down");
    }

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
