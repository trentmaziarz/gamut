//! gamut-media is decode and encode. ffmpeg covers containers, H.264, HEVC
//! and audio, with one module of raw FFI calls for the hardware device context
//! because no safe ffmpeg wrapper exposes hwaccel. libheif covers HEIC and AVIF,
//! rawler covers RAW, and the image crate covers JPEG and PNG. cpal plays back
//! and rubato resamples. A frame cache runs per source, and optional 1080p
//! proxies are generated on import for 4K sources.
//!
//! In M1 the [`photo`] module opens JPEG, PNG and HEIC as oriented RGBA,
//! [`export`] writes the JPEG, and the [`fixtures`] module locates the
//! sample photos the tests open. M2 adds [`video`] (H.264 and HEVC frames
//! as NV12 or P010 through NVDEC when a CUDA device answers, else the
//! software decoder), [`audio`] (any audio stream as f32 stereo at a chosen
//! rate), [`hwaccel`] (the one module of raw FFI calls) and [`video_export`] (the
//! Reel writer through h264_nvenc and aac).

pub mod audio;
pub mod export;
pub mod fixtures;
pub mod hwaccel;
pub mod photo;
pub mod video;
pub mod video_export;

pub use ffmpeg_the_third as ffmpeg;
pub use photo::{Photo, PhotoError, SourceSpace, open_photo};
pub use video::{
    DecodedFrame, Decoder, FramePlanes, PlaneFormat, RawFrame, Transfer, VideoColour, VideoError,
    VideoFrame, VideoSource, YuvSpace,
};

/// Initialises ffmpeg once. Every open path calls it; calling it again is
/// free.
pub fn init_ffmpeg() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        ffmpeg::init().expect("ffmpeg initialises");
        ffmpeg::log::set_level(ffmpeg::log::Level::Warning);
    });
}
