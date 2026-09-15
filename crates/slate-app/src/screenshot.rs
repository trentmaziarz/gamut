//! The `--screenshot` path: render the test image without a window and
//! write it as a PNG. It runs the same slate-gpu passes the Viewer uses,
//! through the headless context, so CI and scripts can see the picture.

use std::error::Error;
use std::fmt;
use std::path::Path;

use slate_gpu::{Headless, Readback, TestImage};

/// The screenshot size: the 4:5 feed post at 1080 wide.
pub const SIZE: (u32, u32) = (1080, 1350);

/// Why a screenshot could not be written.
#[derive(Debug)]
pub enum ScreenshotError {
    /// No adapter answered, neither hardware nor the fallback.
    NoAdapter,
    /// The PNG could not be encoded or written.
    Image(image::ImageError),
}

impl fmt::Display for ScreenshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScreenshotError::NoAdapter => f.write_str("no GPU adapter is available"),
            ScreenshotError::Image(error) => write!(f, "could not write the PNG: {error}"),
        }
    }
}

impl Error for ScreenshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ScreenshotError::NoAdapter => None,
            ScreenshotError::Image(error) => Some(error),
        }
    }
}

/// Renders the test image at [`SIZE`] and writes it to `path` as a PNG.
pub fn write(path: &Path) -> Result<(), ScreenshotError> {
    let gpu = Headless::new().ok_or(ScreenshotError::NoAdapter)?;
    log::info!("adapter: {}", gpu.describe());
    let (width, height) = SIZE;
    let image = TestImage::new(&gpu.device, width, height);
    image.render(&gpu.device, &gpu.queue);
    let pixels =
        Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, image.view(), width, height);
    let buffer = image::RgbaImage::from_raw(width, height, pixels)
        .expect("readback returns width times height times four bytes");
    buffer.save(path).map_err(ScreenshotError::Image)
}
