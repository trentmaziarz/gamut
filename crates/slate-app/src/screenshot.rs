//! The `--screenshot` path: render without a window and write a PNG. With
//! a photo it runs the same develop graph the Viewer uses, through the
//! headless context, so CI and scripts can see the picture. Without one it
//! renders the M0 test image.

use std::error::Error;
use std::fmt;
use std::path::Path;

use slate_core::{Crop, CropRect, Sidecar};
use slate_gpu::develop::render_size_for_crop;
use slate_gpu::{Develop, Headless, Readback, TestImage};
use slate_media::{PhotoError, open_photo};

use crate::sidecar;

/// The screenshot size: the 4:5 feed post at 1080 wide.
pub const SIZE: (u32, u32) = (1080, 1350);

/// Why a screenshot could not be written.
#[derive(Debug)]
pub enum ScreenshotError {
    /// No adapter answered, neither hardware nor the fallback.
    NoAdapter,
    /// The PNG could not be encoded or written.
    Image(image::ImageError),
    /// The photo could not be opened.
    Photo(PhotoError),
    /// The edit file could not be read.
    Edit(String),
}

impl fmt::Display for ScreenshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScreenshotError::NoAdapter => f.write_str("no GPU adapter is available"),
            ScreenshotError::Image(error) => write!(f, "could not write the PNG: {error}"),
            ScreenshotError::Photo(error) => write!(f, "{error}"),
            ScreenshotError::Edit(error) => write!(f, "{error}"),
        }
    }
}

impl Error for ScreenshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ScreenshotError::Image(error) => Some(error),
            ScreenshotError::Photo(error) => Some(error),
            ScreenshotError::NoAdapter | ScreenshotError::Edit(_) => None,
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
    save_png(path, width, height, pixels)
}

/// Opens `photo`, applies its sidecar (or the one at `edit`), renders the
/// crop at [`SIZE`] and writes it to `out` as a PNG.
pub fn write_developed(
    photo: &Path,
    edit: Option<&Path>,
    out: &Path,
) -> Result<(), ScreenshotError> {
    let gpu = Headless::new().ok_or(ScreenshotError::NoAdapter)?;
    log::info!("adapter: {}", gpu.describe());
    let picture = open_photo(photo).map_err(ScreenshotError::Photo)?;
    let sidecar = match edit {
        Some(path) => sidecar::load_from(path).map_err(ScreenshotError::Edit)?,
        None => sidecar::load(photo).unwrap_or_default(),
    };
    let crop = crop_for(&sidecar, picture.width, picture.height);
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&picture);
    let render_size = render_size_for_crop(crop.rect, SIZE);
    let view = develop
        .render(&sidecar.edit, crop.rect, render_size, SIZE)
        .expect("the source was set");
    let (width, height) = SIZE;
    let pixels = Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, view, width, height);
    save_png(out, width, height, pixels)
}

/// The crop a sidecar asks for. A sidecar whose rectangle is the whole
/// photo (the default) is fitted to the photo at its aspect.
pub fn crop_for(sidecar: &Sidecar, width: u32, height: u32) -> Crop {
    if sidecar.crop.rect == CropRect::FULL {
        Crop::fitted(sidecar.crop.aspect, width, height)
    } else {
        sidecar.crop
    }
}

fn save_png(path: &Path, width: u32, height: u32, pixels: Vec<u8>) -> Result<(), ScreenshotError> {
    let buffer = image::RgbaImage::from_raw(width, height, pixels)
        .expect("readback returns width times height times four bytes");
    buffer.save(path).map_err(ScreenshotError::Image)
}
