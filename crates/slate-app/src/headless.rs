//! What the `--screenshot` and `--export` paths share: the headless device,
//! the opened photo and the sidecar that applies to it.

use std::error::Error;
use std::fmt;
use std::path::Path;

use slate_core::{Crop, CropRect, Sidecar};
use slate_gpu::Headless;
use slate_media::export::ExportError;
use slate_media::{Photo, PhotoError, VideoError, open_photo};

use crate::presets;
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

/// Where a headless render takes its edit from, in the order applied.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EditSource<'a> {
    /// A sidecar file; without one, the sidecar next to the photo.
    pub edit: Option<&'a Path>,
    /// A named version of that sidecar in place of its working state.
    pub version: Option<&'a str>,
    /// A look preset applied over the edit.
    pub look: Option<&'a Path>,
}

/// A device, a photo and the edit to apply to it.
pub struct Prepared {
    pub gpu: Headless,
    pub photo: Photo,
    pub sidecar: Sidecar,
}

impl Prepared {
    /// Opens the device and `photo`, then reads the sidecar the source
    /// names, or the one next to the photo, or the neutral edit; takes the
    /// named version out of it when one is asked for; and applies the look
    /// preset over the result.
    pub fn open(photo: &Path, source: EditSource) -> Result<Self, HeadlessError> {
        let mut sidecar = match source.edit {
            Some(path) => sidecar::load_from(path).map_err(HeadlessError::Edit)?,
            None => sidecar::load(photo).unwrap_or_default(),
        };
        if let Some(name) = source.version {
            let version = sidecar
                .version(name)
                .map_err(|error| HeadlessError::Edit(error.to_string()))?
                .clone();
            sidecar.edit = version.edit;
            sidecar.crop = version.crop;
        }
        if let Some(path) = source.look {
            presets::load(path)
                .map_err(HeadlessError::Edit)?
                .apply(&mut sidecar.edit);
        }
        let gpu = Headless::new().ok_or(HeadlessError::NoAdapter)?;
        log::info!("adapter: {}", gpu.describe());
        let picture = open_photo(photo).map_err(HeadlessError::Photo)?;
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

/// The index of the mask `--show-mask` names, or the masks that exist.
pub fn mask_named(sidecar: &Sidecar, name: &str) -> Result<usize, HeadlessError> {
    let masks = &sidecar.edit.masks;
    slate_core::mask::find(masks, name).ok_or_else(|| {
        let name = name.trim();
        HeadlessError::Edit(if masks.is_empty() {
            format!("no mask named {name}: this edit has no masks")
        } else {
            let names: Vec<&str> = masks.iter().map(|m| m.name.as_str()).collect();
            format!("no mask named {name}: the masks are {}", names.join(", "))
        })
    })
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
