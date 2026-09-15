//! slate-media is decode and encode. ffmpeg covers containers, H.264, HEVC
//! and audio, with one unsafe module for the hardware device context because
//! no safe ffmpeg wrapper exposes hwaccel. libheif covers HEIC and AVIF,
//! rawler covers RAW, and the image crate covers JPEG and PNG. cpal plays back
//! and rubato resamples. A frame cache runs per source, and optional 1080p
//! proxies are generated on import for 4K sources.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        // The crate compiles and its test harness runs. Real tests arrive
        // with the milestone that fills the crate.
    }
}
