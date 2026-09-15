//! slate-media is decode and encode. ffmpeg covers containers, H.264, HEVC
//! and audio, with one unsafe module for the hardware device context because
//! no safe ffmpeg wrapper exposes hwaccel. libheif covers HEIC and AVIF,
//! rawler covers RAW, and the image crate covers JPEG and PNG. cpal plays back
//! and rubato resamples. A frame cache runs per source, and optional 1080p
//! proxies are generated on import for 4K sources.
//!
//! In M1 the [`photo`] module opens JPEG, PNG and HEIC as oriented RGBA,
//! [`export`] writes the JPEG, and the [`fixtures`] module locates the
//! sample photos the tests open.

pub mod export;
pub mod fixtures;
pub mod photo;

pub use photo::{Photo, PhotoError, SourceSpace, open_photo};
