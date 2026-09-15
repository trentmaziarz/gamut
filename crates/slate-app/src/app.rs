//! The window: a menu bar and an egui_dock layout of three tabs over the
//! one wgpu device that eframe created. The Viewer tab draws the developed
//! photo, the Adjust tab holds the Basic sliders and the crop.

use std::error::Error;
use std::fmt;
use std::path::PathBuf;

use eframe::CreationContext;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use slate_core::{Crop, PhotoEdit};
use slate_media::open_photo;

use crate::adjust;
use crate::viewer::Viewer;

/// The window title.
pub const WINDOW_TITLE: &str = "slate";

/// The window size at first launch, in points.
pub const WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];

/// The extensions the open dialog offers.
pub const PHOTO_EXTENSIONS: [&str; 5] = ["jpg", "jpeg", "png", "heic", "heif"];

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

/// The photo that is open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenPhoto {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
}

/// The editing state the tabs share: the open photo, its edit and crop,
/// and what needs doing about them.
#[derive(Debug, Default)]
pub struct Session {
    pub photo: Option<OpenPhoto>,
    pub edit: PhotoEdit,
    pub crop: Crop,
    pub grid_guide: bool,
    /// The Viewer must run the develop graph again.
    pub develop_dirty: bool,
    /// One line about the last thing that went wrong, shown in Adjust.
    pub status: Option<String>,
}

impl Session {
    /// A slider or the crop moved: the picture and the sidecar are stale.
    pub fn mark_edited(&mut self) {
        self.develop_dirty = true;
    }
}

/// The application state eframe drives.
pub struct SlateApp {
    dock: DockState<Tab>,
    viewer: Viewer,
    session: Session,
}

impl SlateApp {
    /// Builds the app on the device eframe created, and opens `photo` when
    /// one was named. Never creates a second device.
    pub fn new(cc: &CreationContext<'_>, photo: Option<PathBuf>) -> Result<Self, NoRenderState> {
        let render_state = cc.wgpu_render_state.clone().ok_or(NoRenderState)?;
        let info = render_state.adapter.get_info();
        log::info!("adapter: {} ({:?})", info.name, info.backend);
        let mut app = Self {
            dock: layout(),
            viewer: Viewer::new(render_state),
            session: Session::default(),
        };
        if let Some(path) = photo {
            app.open(path);
        }
        Ok(app)
    }

    /// Opens a photo into the viewer. A failure is reported in the status
    /// line and the previous photo stays.
    pub fn open(&mut self, path: PathBuf) {
        match open_photo(&path) {
            Ok(photo) => {
                self.viewer.set_photo(&photo);
                self.session.crop =
                    Crop::fitted(self.session.crop.aspect, photo.width, photo.height);
                self.session.edit = PhotoEdit::default();
                self.session.photo = Some(OpenPhoto {
                    path,
                    width: photo.width,
                    height: photo.height,
                });
                self.session.status = None;
                self.session.develop_dirty = true;
            }
            Err(error) => {
                log::error!("{error}");
                self.session.status = Some(error.to_string());
            }
        }
    }

    fn menu(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open...").clicked()
                    && let Some(path) = pick_photo()
                {
                    self.open(path);
                }
            });
        });
    }
}

/// The open dialog, filtered to the photo formats slate reads.
fn pick_photo() -> Option<PathBuf> {
    rfd::FileDialog::new()
        .add_filter("Photos", &PHOTO_EXTENSIONS)
        .pick_file()
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
        let dropped = ui
            .ctx()
            .input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()));
        if let Some(path) = dropped {
            self.open(path);
        }
        self.menu(ui);
        let style = Style::from_egui(ui.style().as_ref());
        let mut tabs = Tabs {
            viewer: &mut self.viewer,
            session: &mut self.session,
        };
        DockArea::new(&mut self.dock)
            .style(style)
            .show_close_buttons(false)
            .show_leaf_close_all_buttons(false)
            .show_leaf_collapse_buttons(false)
            .show_inside(ui, &mut tabs);
    }
}

struct Tabs<'a> {
    viewer: &'a mut Viewer,
    session: &'a mut Session,
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
            Tab::Viewer => self.viewer.ui(ui, self.session),
            Tab::Adjust => adjust::ui(ui, self.session),
            Tab::Timeline => {
                ui.label("Timeline arrives in M2");
            }
        }
    }

    fn closeable(&mut self, _tab: &mut Tab) -> bool {
        false
    }
}
