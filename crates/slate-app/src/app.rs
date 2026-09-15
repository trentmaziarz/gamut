//! The window: a menu bar and an egui_dock layout of three tabs over the
//! one wgpu device that eframe created. The Viewer tab draws the developed
//! photo, the Adjust tab holds the Basic sliders and the crop, File > Export
//! opens the export dialog, and the edit is saved as a sidecar next to the
//! photo half a second after the last change and on close.

use std::error::Error;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use eframe::CreationContext;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use slate_core::{Crop, ExportPreset, PhotoEdit, Sidecar};
use slate_media::export::write_jpeg;
use slate_media::open_photo;

use crate::export::{crop_for_preset, proposed_name};
use crate::headless::crop_for;
use crate::viewer::Viewer;
use crate::{adjust, sidecar};

/// The window title.
pub const WINDOW_TITLE: &str = "slate";

/// The window size at first launch, in points.
pub const WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];

/// The extensions the open dialog offers.
pub const PHOTO_EXTENSIONS: [&str; 5] = ["jpg", "jpeg", "png", "heic", "heif"];

/// How long after the last change the sidecar is written.
pub const SIDECAR_DELAY: Duration = Duration::from_millis(500);

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
    /// When the edit or crop last changed and the sidecar has not been
    /// written since.
    pub changed_at: Option<Instant>,
    /// One line about the last thing that happened, shown in Adjust.
    pub status: Option<String>,
}

impl Session {
    /// A slider or the crop moved: the picture and the sidecar are stale.
    pub fn mark_edited(&mut self) {
        self.develop_dirty = true;
        self.changed_at = Some(Instant::now());
    }

    /// Writes the sidecar next to the photo when a change is pending.
    fn save_sidecar(&mut self) {
        let Some(photo) = &self.photo else {
            self.changed_at = None;
            return;
        };
        if self.changed_at.take().is_none() {
            return;
        }
        if let Err(error) = sidecar::save(&photo.path, &Sidecar::new(self.edit, self.crop)) {
            log::error!("could not save the sidecar: {error}");
            self.status = Some(format!("Could not save the sidecar: {error}"));
        }
    }
}

/// The application state eframe drives.
pub struct SlateApp {
    dock: DockState<Tab>,
    viewer: Viewer,
    session: Session,
    /// The export dialog, when open, with the preset it has selected.
    export_dialog: Option<ExportPreset>,
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
            export_dialog: None,
        };
        if let Some(path) = photo {
            app.open(path);
        }
        Ok(app)
    }

    /// Opens a photo into the viewer, with its sidecar when one is next to
    /// it. A failure is reported in the status line and the previous photo
    /// stays.
    pub fn open(&mut self, path: PathBuf) {
        self.session.save_sidecar();
        match open_photo(&path) {
            Ok(photo) => {
                self.viewer.set_photo(&photo);
                let saved = sidecar::load(&path);
                let (edit, crop) = match saved {
                    Some(sidecar) => (sidecar.edit, crop_for(&sidecar, photo.width, photo.height)),
                    None => (
                        PhotoEdit::default(),
                        Crop::fitted(self.session.crop.aspect, photo.width, photo.height),
                    ),
                };
                self.session.edit = edit;
                self.session.crop = crop;
                self.session.photo = Some(OpenPhoto {
                    path,
                    width: photo.width,
                    height: photo.height,
                });
                self.session.status = None;
                self.session.develop_dirty = true;
                self.session.changed_at = None;
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
                let has_photo = self.session.photo.is_some();
                if ui
                    .add_enabled(has_photo, egui::Button::new("Export..."))
                    .clicked()
                {
                    self.export_dialog = Some(ExportPreset::for_aspect(self.session.crop.aspect));
                }
            });
        });
    }

    /// The export dialog: four presets and a Save button.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut preset) = self.export_dialog else {
            return;
        };
        let mut open = true;
        let mut save = false;
        let mut cancel = false;
        egui::Window::new("Export")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                for candidate in ExportPreset::ALL {
                    ui.radio_value(&mut preset, candidate, candidate.label());
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save...").clicked() {
                        save = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        self.export_dialog = (open && !cancel).then_some(preset);
        if save {
            self.export(preset);
        }
    }

    /// Asks where to save, renders the export and writes the JPEG.
    fn export(&mut self, preset: ExportPreset) {
        let Some(photo) = self.session.photo.clone() else {
            return;
        };
        let proposed = proposed_name(&photo.path, preset);
        let mut dialog = rfd::FileDialog::new()
            .add_filter("JPEG", &["jpg"])
            .set_file_name(proposed.to_string_lossy());
        if let Some(dir) = photo.path.parent() {
            dialog = dialog.set_directory(dir);
        }
        let Some(out) = dialog.save_file() else {
            return;
        };
        let crop = crop_for_preset(self.session.crop, preset, photo.width, photo.height);
        let (width, height) = preset.size();
        let result = self
            .viewer
            .export(&self.session.edit, crop, preset)
            .ok_or_else(|| "no photo to export".to_string())
            .and_then(|pixels| {
                write_jpeg(&pixels, width, height, &out).map_err(|error| error.to_string())
            });
        self.session.status = Some(match result {
            Ok(()) => format!("Exported {}", out.display()),
            Err(error) => error,
        });
        self.session.develop_dirty = true;
        self.export_dialog = None;
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
        self.export_dialog(ui.ctx());

        if let Some(changed_at) = self.session.changed_at {
            let waited = changed_at.elapsed();
            if waited >= SIDECAR_DELAY {
                self.session.save_sidecar();
            } else {
                ui.ctx().request_repaint_after(SIDECAR_DELAY - waited);
            }
        }
    }

    fn on_exit(&mut self) {
        self.session.save_sidecar();
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
