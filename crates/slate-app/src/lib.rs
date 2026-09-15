//! slate-app is the egui application. One window, egui_dock panels, a Photo
//! view and a Video view over the same engine, the phone-frame preview, the
//! export dialog. Startup target is under 2 seconds.
//!
//! In M0 the window holds three docked tabs and the Viewer draws the
//! slate-gpu test image.

pub mod app;

pub use app::{SlateApp, WINDOW_TITLE, native_options};
