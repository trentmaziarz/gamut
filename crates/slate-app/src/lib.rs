//! slate-app is the egui application. One window, egui_dock panels, a Photo
//! view and a Video view over the same engine, the phone-frame preview, the
//! export dialog. Startup target is under 2 seconds.
//!
//! In M1 the window opens a photo three ways (the command line, a dropped
//! file, File > Open), the [`adjust`] tab binds the Basic sliders and the
//! crop, the [`viewer`] draws the developed picture under the phone frame,
//! the edit is saved as a [`sidecar`] next to the photo, and the
//! [`screenshot`] and [`export`] paths render the same picture without a
//! window.

pub mod adjust;
pub mod app;
pub mod export;
pub mod headless;
pub mod screenshot;
pub mod sidecar;
pub mod viewer;

pub use app::{SlateApp, WINDOW_TITLE, native_options};
