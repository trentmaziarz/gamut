//! The window: an egui_dock layout of three tabs over the one wgpu device
//! that eframe created. The Viewer tab draws the slate-gpu test image.

use std::error::Error;
use std::fmt;

use eframe::CreationContext;
use egui::load::SizedTexture;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use egui_wgpu::RenderState;
use egui_wgpu::wgpu;
use slate_core::PhotoEdit;
use slate_gpu::TestImage;

/// The window title.
pub const WINDOW_TITLE: &str = "slate";

/// The window size at first launch, in points.
pub const WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];

/// The viewer keeps the 4:5 feed-post shape, as width to height.
pub const VIEWER_ASPECT: [f32; 2] = [4.0, 5.0];

/// The size the viewer texture starts at, before the tab reports its size.
const INITIAL_VIEWER_SIZE: (u32, u32) = (1080, 1350);

/// The eframe options: the wgpu renderer, the title and the launch size.
pub fn native_options() -> eframe::NativeOptions {
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(WINDOW_TITLE)
            .with_inner_size(WINDOW_SIZE),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    }
}

/// eframe gave the app no wgpu render state, so there is no device to
/// share with the engine.
#[derive(Debug)]
pub struct NoRenderState;

impl fmt::Display for NoRenderState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("eframe gave the app no wgpu render state")
    }
}

impl Error for NoRenderState {}

/// The three tabs of M0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tab {
    Viewer,
    Adjust,
    Timeline,
}

impl Tab {
    fn title(self) -> &'static str {
        match self {
            Tab::Viewer => "Viewer",
            Tab::Adjust => "Adjust",
            Tab::Timeline => "Timeline",
        }
    }
}

/// The application state eframe drives.
pub struct SlateApp {
    dock: DockState<Tab>,
    viewer: Viewer,
    /// The develop state of the open photo. M1 binds the Adjust panel to it.
    edit: PhotoEdit,
}

impl SlateApp {
    /// Builds the app on the device eframe created. Never creates a second one.
    pub fn new(cc: &CreationContext<'_>) -> Result<Self, NoRenderState> {
        let render_state = cc.wgpu_render_state.clone().ok_or(NoRenderState)?;
        let info = render_state.adapter.get_info();
        log::info!("adapter: {} ({:?})", info.name, info.backend);
        Ok(Self {
            dock: layout(),
            viewer: Viewer::new(render_state),
            edit: PhotoEdit::default(),
        })
    }
}

/// Viewer in the centre, Adjust on the right, Timeline along the bottom.
fn layout() -> DockState<Tab> {
    let mut dock = DockState::new(vec![Tab::Viewer]);
    let tree = dock.main_surface_mut();
    let [viewer, _adjust] = tree.split_right(NodeIndex::root(), 0.75, vec![Tab::Adjust]);
    tree.split_below(viewer, 0.7, vec![Tab::Timeline]);
    dock
}

impl eframe::App for SlateApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let style = Style::from_egui(ui.style().as_ref());
        let mut tabs = Tabs {
            viewer: &mut self.viewer,
            edit: &self.edit,
        };
        DockArea::new(&mut self.dock)
            .style(style)
            .show_close_buttons(false)
            .show_inside(ui, &mut tabs);
    }
}

struct Tabs<'a> {
    viewer: &'a mut Viewer,
    edit: &'a PhotoEdit,
}

impl TabViewer for Tabs<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(tab.title())
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        match tab {
            Tab::Viewer => self.viewer.ui(ui),
            Tab::Adjust => {
                ui.label("Develop panel arrives in M1");
                if *self.edit == PhotoEdit::default() {
                    ui.weak("Every slider sits at neutral.");
                }
            }
            Tab::Timeline => {
                ui.label("Timeline arrives in M2");
            }
        }
    }

    fn closeable(&mut self, _tab: &mut Tab) -> bool {
        false
    }
}

/// The offscreen test image and the egui texture that shows it.
struct Viewer {
    render_state: RenderState,
    image: TestImage,
    texture_id: egui::TextureId,
}

impl Viewer {
    fn new(render_state: RenderState) -> Self {
        let (width, height) = INITIAL_VIEWER_SIZE;
        let image = TestImage::new(&render_state.device, width, height);
        image.render(&render_state.device, &render_state.queue);
        let texture_id = render_state.renderer.write().register_native_texture(
            &render_state.device,
            image.view(),
            wgpu::FilterMode::Linear,
        );
        Self {
            render_state,
            image,
            texture_id,
        }
    }

    /// Draws the image at the largest 4:5 size that fits the tab. The
    /// texture is re-rendered and re-registered only when that size changes.
    fn ui(&mut self, ui: &mut egui::Ui) {
        let points = fit_aspect(ui.available_size(), VIEWER_ASPECT);
        let scale = ui.pixels_per_point();
        let wanted = (
            (points.x * scale).round().max(1.0) as u32,
            (points.y * scale).round().max(1.0) as u32,
        );
        if wanted != self.image.size() {
            let device = &self.render_state.device;
            self.image.resize(device, wanted.0, wanted.1);
            self.image.render(device, &self.render_state.queue);
            self.render_state
                .renderer
                .write()
                .update_egui_texture_from_wgpu_texture(
                    device,
                    self.image.view(),
                    wgpu::FilterMode::Linear,
                    self.texture_id,
                );
        }
        ui.centered_and_justified(|ui| {
            ui.add(egui::Image::from_texture(SizedTexture::new(
                self.texture_id,
                points,
            )));
        });
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
    use super::{VIEWER_ASPECT, fit_aspect};

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
}
