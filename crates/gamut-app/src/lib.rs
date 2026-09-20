//! gamut-app is the egui application. One window, egui_dock panels, a Photo
//! view and a Video view over the same engine, the phone-frame preview, the
//! export dialog. Startup target is under 2 seconds.
//!
//! In M1 the window opens a photo three ways (the command line, a dropped
//! file, File > Open), the [`adjust`] tab binds the Basic sliders and the
//! crop, the [`viewer`] draws the developed picture under the phone frame,
//! the edit is saved as a [`sidecar`] next to the photo, and the
//! [`screenshot`] and [`export`] paths render the same picture without a
//! window.
//!
//! In M2 a video or a .gamut file opens a [`project`], the [`player`]
//! decodes and clocks it, the [`timeline_tab`] cuts the one track, the
//! Viewer draws the frame under the playhead, and [`reel`] writes the
//! 1080x1920 Reel.

pub mod adjust;
pub mod app;
pub mod brush_tool;
pub mod curve_editor;
pub mod export;
pub mod headless;
pub mod mask_handles;
pub mod mask_panel;
pub mod player;
pub mod presets;
pub mod project;
pub mod reel;
pub mod screenshot;
pub mod sidecar;
pub mod timeline_tab;
pub mod undo;
pub mod view;
pub mod viewer;
pub mod wheel;

pub use app::{GamutApp, WINDOW_TITLE, native_options};
