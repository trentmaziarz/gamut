//! What the `--screenshot` and `--export` paths share: the headless device,
//! the opened photo and the sidecar that applies to it.

use std::error::Error;
use std::fmt;
use std::path::Path;

use slate_core::{Crop, CropRect, Sidecar};
use slate_gpu::Headless;
use slate_media::export::ExportError;
use slate_media::{Photo, PhotoError, VideoError, open_photo};

use crate::sidecar;

/// Why a headless render could not be written.
#[derive(Debug)]
pub enum HeadlessError {
    /// No adapter answered, neither hardware nor the fallback.
    NoAdapter,
    /// The PNG could not be encoded or written.
    Image(image::ImageError),
    /// The JPEG could not be encoded or written.
    Jpeg(ExportError),
    /// The photo could not be opened.
    Photo(PhotoError),
    /// The edit file could not be read, or a project is not usable.
    Edit(String),
    /// The video could not be opened or decoded.
    Video(VideoError),
}

impl fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeadlessError::NoAdapter => f.write_str("no GPU adapter is available"),
            HeadlessError::Image(error) => write!(f, "could not write the PNG: {error}"),
            HeadlessError::Jpeg(error) => write!(f, "{error}"),
            HeadlessError::Photo(error) => write!(f, "{error}"),
            HeadlessError::Edit(error) => write!(f, "{error}"),
            HeadlessError::Video(error) => write!(f, "{error}"),
        }
    }
}

impl Error for HeadlessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            HeadlessError::Image(error) => Some(error),
            HeadlessError::Jpeg(error) => Some(error),
            HeadlessError::Photo(error) => Some(error),
            HeadlessError::Video(error) => Some(error),
            HeadlessError::NoAdapter | HeadlessError::Edit(_) => None,
        }
    }
}

/// A device, a photo and the edit to apply to it.
pub struct Prepared {
    pub gpu: Headless,
    pub photo: Photo,
    pub sidecar: Sidecar,
}

impl Prepared {
    /// Opens the device and `photo`, then reads the sidecar at `edit`, or
    /// the one next to the photo, or the neutral edit.
    pub fn open(photo: &Path, edit: Option<&Path>) -> Result<Self, HeadlessError> {
        let gpu = Headless::new().ok_or(HeadlessError::NoAdapter)?;
        log::info!("adapter: {}", gpu.describe());
        let picture = open_photo(photo).map_err(HeadlessError::Photo)?;
        let sidecar = match edit {
            Some(path) => sidecar::load_from(path).map_err(HeadlessError::Edit)?,
            None => sidecar::load(photo).unwrap_or_default(),
        };
        Ok(Prepared {
            gpu,
            photo: picture,
            sidecar,
        })
    }

    /// The crop the sidecar asks for on this photo.
    pub fn crop(&self) -> Crop {
        crop_for(&self.sidecar, self.photo.width, self.photo.height)
    }
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
