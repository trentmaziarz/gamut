//! The window: a menu bar and an egui_dock layout of three tabs over the
//! one wgpu device that eframe created. A session holds either a photo or
//! a project. The Viewer tab draws the developed photo, or the frame under
//! the playhead, the Adjust tab holds the Basic sliders and the crop, the
//! Timeline tab holds the one track, File > Export opens the export
//! dialog, and the edit is saved half a second after the last change and
//! on close: as a sidecar next to a photo, as the .slate file of a project.

use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::CreationContext;
use egui_dock::{DockArea, DockState, NodeIndex, Style, TabViewer};
use slate_core::{
    Crop, CropAspect, ExportPreset, PhotoEdit, Project, Sidecar, Track, VersionError,
};
use slate_media::export::write_jpeg;
use slate_media::open_photo;
use slate_media::video::is_video_path;

use crate::export::{crop_for_preset, proposed_name};
use crate::headless::crop_for;
use crate::player::{MediaInfo, Player};
use crate::project::{self as project_file, LoadedProject};
use crate::viewer::Viewer;
use crate::{adjust, sidecar, timeline_tab};

/// The window title.
pub const WINDOW_TITLE: &str = "slate";

/// The window size at first launch, in points.
pub const WINDOW_SIZE: [f32; 2] = [1280.0, 800.0];

/// The extensions the open dialog offers for photos.
pub const PHOTO_EXTENSIONS: [&str; 5] = ["jpg", "jpeg", "png", "heic", "heif"];

/// The extensions the open dialog offers for videos.
pub const VIDEO_EXTENSIONS: [&str; 3] = ["mp4", "mov", "m4v"];

/// How long after the last change the sidecar or project is written.
pub const SIDECAR_DELAY: Duration = Duration::from_millis(500);

/// The eframe options: the wgpu renderer, the title, the launch size and
/// the device features the video planes need.
pub fn native_options() -> eframe::NativeOptions {
    let mut setup = egui_wgpu::WgpuSetupCreateNew::without_display_handle();
    let base = setup.device_descriptor.clone();
    setup.device_descriptor = Arc::new(move |adapter| {
        let mut descriptor = base(adapter);
        descriptor.required_features |= slate_gpu::video::wanted_features(adapter);
        descriptor
    });
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(WINDOW_TITLE)
            .with_inner_size(WINDOW_SIZE),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: egui_wgpu::WgpuSetup::CreateNew(setup),
            ..Default::default()
        },
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

/// The three tabs.
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

/// The project that is open: its file, its media, the track being edited
/// and the player that plays it.
pub struct OpenProject {
    /// The .slate file.
    pub path: PathBuf,
    /// The folder the media paths are relative to.
    pub dir: PathBuf,
    pub media: Vec<MediaInfo>,
    /// The media list as saved, so the file keeps its paths.
    pub project: Project,
    pub track: Track,
    pub player: Player,
    pub selected: Option<usize>,
    /// The track changed and the session has not been told yet.
    pub track_dirty: bool,
}

impl OpenProject {
    /// The frame rate of the first media, for timecodes.
    pub fn frame_rate(&self) -> f64 {
        self.media.first().map(|m| m.frame_rate).unwrap_or(30.0)
    }

    /// The display size of the first media.
    pub fn size(&self) -> (u32, u32) {
        self.media
            .first()
            .map(|m| (m.width, m.height))
            .unwrap_or((1080, 1920))
    }

    fn track_changed(&mut self) {
        self.player.set_track(self.track.clone());
        self.track_dirty = true;
    }

    /// Splits the clip under the playhead and selects the second half.
    pub fn split_at_playhead(&mut self) {
        let at = self.player.position();
        if let Some(index) = self.track.split_at(at) {
            self.selected = Some(index);
            self.track_changed();
        }
    }

    /// Ripple-deletes the selected clip.
    pub fn delete_selected(&mut self) {
        let Some(index) = self.selected.take() else {
            return;
        };
        if self.track.ripple_delete(index).is_some() {
            self.track_changed();
        }
    }

    /// Moves a clip's in point (`is_in`) or out point by `seconds`.
    pub fn trim(&mut self, index: usize, is_in: bool, seconds: f64) {
        let Some(clip) = self.track.clips.get(index).copied() else {
            return;
        };
        let media_duration = self
            .media
            .get(clip.media)
            .map(|m| m.duration)
            .unwrap_or(clip.source_out);
        if is_in {
            self.track.trim_in(index, clip.source_in + seconds);
        } else {
            self.track
                .trim_out(index, clip.source_out + seconds, media_duration);
        }
        if self.track.clips[index] != clip {
            self.selected = Some(index);
            self.track_changed();
        }
    }

    /// The project as it should be saved, with the live edit and crop.
    pub fn snapshot(&self, edit: &PhotoEdit, crop: Crop) -> Project {
        Project {
            track: self.track.clone(),
            crop,
            edit: edit.clone(),
            ..self.project.clone()
        }
    }
}

/// What the person answered when a switch met unsaved work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwitchAnswer {
    /// Save the work into the active version, then switch.
    Save,
    /// Switch and let the work go.
    Discard,
    /// Stay on the working state.
    Cancel,
}

/// The editing state the tabs share: the open photo or project, its edit
/// and crop, and what needs doing about them.
#[derive(Default)]
pub struct Session {
    pub photo: Option<OpenPhoto>,
    pub project: Option<OpenProject>,
    pub edit: PhotoEdit,
    pub crop: Crop,
    /// The named versions of the open photo and the one the working state
    /// came from.
    pub versions: Vec<slate_core::NamedVersion>,
    pub active_version: Option<String>,
    /// What the Adjust tab remembers that is not part of the edit.
    pub adjust: crate::adjust::AdjustState,
    /// A small copy of the open photo that a range mask is picked from.
    pub pick: Option<crate::mask_handles::PickImage>,
    pub grid_guide: bool,
    /// The Viewer must run the develop graph again.
    pub develop_dirty: bool,
    /// When the edit, crop or track last changed and the file has not been
    /// written since.
    pub changed_at: Option<Instant>,
    /// One line about the last thing that happened, shown in Adjust.
    pub status: Option<String>,
    /// Playback is running, so the window must draw again at once.
    pub repaint_wanted: bool,
}

impl Session {
    /// A slider, the crop or the track moved: the picture and the file are
    /// stale.
    pub fn mark_edited(&mut self) {
        self.develop_dirty = true;
        self.changed_at = Some(Instant::now());
    }

    /// The size of what is open, for the crop buttons.
    pub fn source_size(&self) -> Option<(u32, u32)> {
        match (&self.photo, &self.project) {
            (Some(photo), _) => Some((photo.width, photo.height)),
            (None, Some(project)) => Some(project.size()),
            (None, None) => None,
        }
    }

    /// One line naming what is open.
    pub fn open_name(&self) -> Option<String> {
        let path = match (&self.photo, &self.project) {
            (Some(photo), _) => &photo.path,
            (None, Some(project)) => &project.path,
            (None, None) => return None,
        };
        path.file_name().map(|n| n.to_string_lossy().into_owned())
    }

    /// The open photo's state as its sidecar holds it.
    pub fn sidecar(&self) -> Sidecar {
        Sidecar {
            versions: self.versions.clone(),
            active_version: self.active_version.clone(),
            ..Sidecar::new(self.edit.clone(), self.crop)
        }
    }

    /// Runs a version operation of [`Sidecar`] on the session's state and
    /// takes the result back. On success the picture and the file are stale.
    pub fn with_versions<E>(
        &mut self,
        operation: impl FnOnce(&mut Sidecar) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut sidecar = self.sidecar();
        operation(&mut sidecar)?;
        self.edit = sidecar.edit;
        self.crop = sidecar.crop;
        self.versions = sidecar.versions;
        self.active_version = sidecar.active_version;
        self.mark_edited();
        Ok(())
    }

    /// Takes the state a sidecar holds: the working edit and crop, the
    /// versions and the active one.
    pub fn take_sidecar(&mut self, sidecar: Sidecar) {
        self.edit = sidecar.edit;
        self.crop = sidecar.crop;
        self.versions = sidecar.versions;
        self.active_version = sidecar.active_version;
    }

    /// Whether the working state holds work its active version does not.
    pub fn version_is_dirty(&self) -> bool {
        self.sidecar().is_dirty()
    }

    /// The Switch to button. On a clean working state the version comes in
    /// at once. On a dirty one nothing changes yet: the switch waits in
    /// `adjust.pending_switch` for [`Session::answer_switch`].
    pub fn request_switch(&mut self, name: &str) -> Result<(), VersionError> {
        if self.version_is_dirty() {
            self.sidecar().version(name)?;
            self.adjust.pending_switch = Some(name.to_string());
            // What the last action said is not about this one.
            self.status = None;
            return Ok(());
        }
        self.with_versions(|sidecar| sidecar.switch_to(name))?;
        self.status = Some(format!("Switched to {name}."));
        Ok(())
    }

    /// The answer to the prompt a dirty switch raised. Save keeps the work
    /// in the active version first, Discard lets it go, Cancel stays put.
    pub fn answer_switch(&mut self, answer: SwitchAnswer) -> Result<(), VersionError> {
        let Some(target) = self.adjust.pending_switch.take() else {
            return Ok(());
        };
        let active = self.active_version.clone();
        match answer {
            SwitchAnswer::Cancel => {}
            SwitchAnswer::Save => {
                self.with_versions(|sidecar| {
                    if let Some(active) = &active {
                        sidecar.update_version(active)?;
                    }
                    sidecar.switch_to(&target)
                })?;
                let kept = active.unwrap_or_default();
                self.status = Some(format!("Saved into {kept}, switched to {target}."));
            }
            SwitchAnswer::Discard => {
                self.with_versions(|sidecar| sidecar.switch_to(&target))?;
                self.status = Some(format!("Discarded the changes, switched to {target}."));
            }
        }
        Ok(())
    }

    /// The Update button of a version row: the working state goes into that
    /// version.
    pub fn update_version(&mut self, name: &str) -> Result<(), VersionError> {
        self.with_versions(|sidecar| sidecar.update_version(name))?;
        self.status = Some(format!("Updated {name}."));
        Ok(())
    }

    /// Writes the sidecar or the project file when a change is pending.
    fn save(&mut self) {
        if self.changed_at.take().is_none() {
            return;
        }
        let result = if let Some(photo) = &self.photo {
            sidecar::save(&photo.path, &self.sidecar()).map(|_| ())
        } else if let Some(project) = &self.project {
            project_file::save(&project.path, &project.snapshot(&self.edit, self.crop))
        } else {
            Ok(())
        };
        if let Err(error) = result {
            log::error!("could not save: {error}");
            self.status = Some(format!("Could not save: {error}"));
        }
    }

    /// Saves now regardless of the timer.
    pub fn save_now(&mut self) {
        self.changed_at = Some(Instant::now());
        self.save();
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
    /// Builds the app on the device eframe created, and opens `path` when
    /// one was named. Never creates a second device.
    pub fn new(cc: &CreationContext<'_>, path: Option<PathBuf>) -> Result<Self, NoRenderState> {
        let render_state = cc.wgpu_render_state.clone().ok_or(NoRenderState)?;
        let info = render_state.adapter.get_info();
        log::info!("adapter: {} ({:?})", info.name, info.backend);
        let mut app = Self {
            dock: layout(),
            viewer: Viewer::new(render_state),
            session: Session::default(),
            export_dialog: None,
        };
        if let Some(path) = path {
            app.open(path);
        }
        Ok(app)
    }

    /// Opens a file by its extension: a photo, a video into a fresh
    /// project, or a project. A failure is reported in the status line and
    /// what was open stays.
    pub fn open(&mut self, path: PathBuf) {
        self.session.save();
        let result = if Project::is_project_path(&path) {
            project_file::load(&path).and_then(|loaded| self.open_project(loaded))
        } else if is_video_path(&path) {
            MediaInfo::probe(&path)
                .map_err(|error| error.to_string())
                .and_then(|info| self.open_project(project_file::for_video(&path, &info)))
        } else {
            self.open_photo(path)
        };
        if let Err(error) = result {
            log::error!("{error}");
            self.session.status = Some(error);
        }
    }

    fn open_photo(&mut self, path: PathBuf) -> Result<(), String> {
        let photo = open_photo(&path).map_err(|error| error.to_string())?;
        self.viewer.set_photo(&photo);
        self.session.pick = Some(crate::mask_handles::PickImage::from_photo(&photo));
        self.session.adjust.select_mask(None);
        let saved = sidecar::load(&path);
        let sidecar = match saved {
            Some(sidecar) => Sidecar {
                crop: crop_for(&sidecar, photo.width, photo.height),
                ..sidecar
            },
            None => Sidecar::new(
                PhotoEdit::default(),
                Crop::fitted(self.session.crop.aspect, photo.width, photo.height),
            ),
        };
        self.session.project = None;
        self.session.take_sidecar(sidecar);
        self.session.adjust.pending_switch = None;
        self.session.photo = Some(OpenPhoto {
            path,
            width: photo.width,
            height: photo.height,
        });
        self.session.status = None;
        self.session.develop_dirty = true;
        self.session.changed_at = None;
        Ok(())
    }

    fn open_project(&mut self, loaded: LoadedProject) -> Result<(), String> {
        let media = project_file::probe_media(&loaded)?;
        if media.is_empty() {
            return Err(format!("{} names no media", loaded.path.display()));
        }
        let track = loaded.project.track.clone();
        let player = Player::new(media.clone(), track.clone());
        let (width, height) = (media[0].width, media[0].height);
        let crop = if loaded.project.crop.rect == slate_core::CropRect::FULL {
            Crop::fitted(CropAspect::Story9x16, width, height)
        } else {
            loaded.project.crop
        };
        self.session.photo = None;
        self.session.pick = None;
        self.session.adjust.select_mask(None);
        self.session.versions.clear();
        self.session.active_version = None;
        self.session.edit = loaded.project.edit.clone();
        self.session.crop = crop;
        self.session.project = Some(OpenProject {
            path: loaded.path,
            dir: loaded.dir,
            media,
            project: loaded.project,
            track,
            player,
            selected: None,
            track_dirty: false,
        });
        self.viewer.clear_video();
        self.session.status = None;
        self.session.develop_dirty = true;
        self.session.changed_at = None;
        if let Some(project) = &mut self.session.project {
            project.player.seek(0.0);
        }
        Ok(())
    }

    fn menu(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open...").clicked()
                    && let Some(path) = pick_file()
                {
                    self.open(path);
                }
                let has_project = self.session.project.is_some();
                if ui
                    .add_enabled(has_project, egui::Button::new("Save As..."))
                    .clicked()
                {
                    self.save_as();
                }
                let has_source = self.session.photo.is_some() || has_project;
                if ui
                    .add_enabled(has_source, egui::Button::new("Export..."))
                    .clicked()
                {
                    self.export_dialog = Some(ExportPreset::for_aspect(self.session.crop.aspect));
                }
            });
        });
    }

    /// Asks for a new .slate path and moves the project there.
    fn save_as(&mut self) {
        let Some(project) = self.session.project.as_mut() else {
            return;
        };
        let mut dialog = rfd::FileDialog::new()
            .add_filter("slate project", &[slate_core::project::EXTENSION])
            .set_file_name(
                project
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        if let Some(dir) = project.path.parent() {
            dialog = dialog.set_directory(dir);
        }
        let Some(out) = dialog.save_file() else {
            return;
        };
        let new_dir = project_file::dir_of(&out);
        project_file::rebase(&mut project.project, &project.dir, &new_dir);
        project.path = out;
        project.dir = new_dir;
        self.session.save_now();
        if self.session.status.is_none()
            && let Some(project) = &self.session.project
        {
            self.session.status = Some(format!("Saved {}", project.path.display()));
        }
    }

    /// The export dialog: the four photo presets, or the one Reel preset
    /// for a project, and a Save button.
    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(mut preset) = self.export_dialog else {
            return;
        };
        let is_project = self.session.project.is_some();
        let mut open = true;
        let mut save = false;
        let mut cancel = false;
        egui::Window::new("Export")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                if is_project {
                    preset = ExportPreset::Story9x16;
                    ui.label("Reel 9:16, 1080 by 1920, H.264 with AAC");
                } else {
                    for candidate in ExportPreset::ALL {
                        ui.radio_value(&mut preset, candidate, candidate.label());
                    }
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
            if is_project {
                self.export_reel();
            } else {
                self.export(preset);
            }
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

    /// Asks where to save and writes the Reel from the project as it is.
    fn export_reel(&mut self) {
        let Some(project) = self.session.project.as_mut() else {
            return;
        };
        project.player.pause();
        let stem = project
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "reel".to_string());
        let mut dialog = rfd::FileDialog::new()
            .add_filter("MP4", &["mp4"])
            .set_file_name(format!("{stem}_reel.mp4"));
        if let Some(dir) = project.path.parent() {
            dialog = dialog.set_directory(dir);
        }
        let Some(out) = dialog.save_file() else {
            return;
        };
        let snapshot = project.snapshot(&self.session.edit, self.session.crop);
        let dir = project.dir.clone();
        let result = crate::reel::export(&snapshot, &dir, &out).map_err(|error| error.to_string());
        self.session.status = Some(match result {
            Ok(seconds) => format!("Exported {} in {seconds:.1} s", out.display()),
            Err(error) => error,
        });
        self.session.develop_dirty = true;
        self.export_dialog = None;
    }
}

/// The open dialog, filtered to the photo, video and project formats slate
/// reads.
fn pick_file() -> Option<PathBuf> {
    let mut all: Vec<&str> = PHOTO_EXTENSIONS.to_vec();
    all.extend(VIDEO_EXTENSIONS);
    all.push(slate_core::project::EXTENSION);
    rfd::FileDialog::new()
        .add_filter("Photos, videos and projects", &all)
        .add_filter("Photos", &PHOTO_EXTENSIONS)
        .add_filter("Videos", &VIDEO_EXTENSIONS)
        .add_filter("slate projects", &[slate_core::project::EXTENSION])
        .pick_file()
}

/// Whether a path is something the window opens.
pub fn opens(path: &Path) -> bool {
    Project::is_project_path(path)
        || is_video_path(path)
        || path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| PHOTO_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
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

        if std::mem::take(&mut self.session.repaint_wanted) {
            ui.ctx().request_repaint();
        }
        if let Some(project) = &self.session.project
            && let Some(error) = project.player.take_error()
        {
            log::error!("{error}");
            self.session.status = Some(error);
        }
        if let Some(changed_at) = self.session.changed_at {
            let waited = changed_at.elapsed();
            if waited >= SIDECAR_DELAY {
                self.session.save();
            } else {
                ui.ctx().request_repaint_after(SIDECAR_DELAY - waited);
            }
        }
    }

    fn on_exit(&mut self) {
        self.session.save();
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
            Tab::Timeline => timeline_tab::ui(ui, self.session),
        }
    }

    fn closeable(&mut self, _tab: &mut Tab) -> bool {
        false
    }
}
