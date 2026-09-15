//! Opening a photo: JPEG and PNG through the image crate, HEIC and HEIF
//! through libheif. The result is oriented 8-bit RGBA plus the colour space
//! the pixels are in.

use std::path::{Path, PathBuf};

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbaImage};
use libheif_rs::{ColorPrimaries, ColorSpace, HeifContext, ImageHandle, LibHeif, RgbChroma};
use thiserror::Error;

pub use slate_color::SourceSpace;

/// A decoded photo in display orientation.
#[derive(Clone, Debug, PartialEq)]
pub struct Photo {
    /// Width after the orientation is applied.
    pub width: u32,
    /// Height after the orientation is applied.
    pub height: u32,
    /// Row-major RGBA, 8 bits per channel, no padding.
    pub rgba8: Vec<u8>,
    /// The colour space the bytes are encoded in.
    pub source: SourceSpace,
    /// Bits per channel in the file, before the conversion to 8.
    pub bit_depth: u8,
    /// Whether the file carried an alpha channel.
    pub has_alpha: bool,
}

impl Photo {
    /// The pixel at `x`, `y` as RGBA.
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let at = ((y * self.width + x) * 4) as usize;
        [
            self.rgba8[at],
            self.rgba8[at + 1],
            self.rgba8[at + 2],
            self.rgba8[at + 3],
        ]
    }

    /// Width over height.
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// Why a photo could not be opened.
#[derive(Debug, Error)]
pub enum PhotoError {
    #[error("could not read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not decode {path}: {source}")]
    Image {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("could not decode {path}: {source}")]
    Heif {
        path: PathBuf,
        #[source]
        source: libheif_rs::HeifError,
    },
    #[error("{path} decoded to {bits} bit planes, not the 8 bit interleaved RGB slate asked for")]
    UnexpectedPlanes { path: PathBuf, bits: u8 },
    #[error("{path} is not a photo format slate opens (jpg, jpeg, png, heic, heif)")]
    Unsupported { path: PathBuf },
}

/// Opens `path` as a photo. The format is chosen by the extension.
pub fn open_photo(path: &Path) -> Result<Photo, PhotoError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match extension.as_str() {
        "heic" | "heif" => open_heif(path),
        "jpg" | "jpeg" | "png" => open_image(path),
        _ => Err(PhotoError::Unsupported {
            path: path.to_path_buf(),
        }),
    }
}

fn open_image(path: &Path) -> Result<Photo, PhotoError> {
    let io = |source| PhotoError::Io {
        path: path.to_path_buf(),
        source,
    };
    let decode = |source| PhotoError::Image {
        path: path.to_path_buf(),
        source,
    };
    let reader = ImageReader::open(path)
        .map_err(io)?
        .with_guessed_format()
        .map_err(io)?;
    let mut decoder = reader.into_decoder().map_err(decode)?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let icc = decoder.icc_profile().ok().flatten();
    let color = decoder.color_type();
    let bit_depth = (color.bits_per_pixel() / u16::from(color.channel_count())) as u8;
    let has_alpha = color.has_alpha();
    let mut image = DynamicImage::from_decoder(decoder).map_err(decode)?;
    image.apply_orientation(orientation);
    let rgba = image.into_rgba8();
    let source = match icc {
        Some(bytes) => slate_color::icc::source_space(&bytes),
        None => {
            log::info!(
                "{} carries no ICC profile; treating it as sRGB",
                path.display()
            );
            SourceSpace::Srgb
        }
    };
    log::info!(
        "opened {} as {}x{} {source:?} through the image crate, orientation {orientation:?}",
        path.display(),
        rgba.width(),
        rgba.height()
    );
    Ok(Photo {
        width: rgba.width(),
        height: rgba.height(),
        rgba8: rgba.into_raw(),
        source,
        bit_depth,
        has_alpha,
    })
}

fn open_heif(path: &Path) -> Result<Photo, PhotoError> {
    let heif = |source| PhotoError::Heif {
        path: path.to_path_buf(),
        source,
    };
    let bytes = std::fs::read(path).map_err(|source| PhotoError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let lib = LibHeif::new();
    let context = HeifContext::read_from_bytes(&bytes).map_err(heif)?;
    let handle = context.primary_image_handle().map_err(heif)?;
    let has_alpha = handle.has_alpha_channel();
    let bit_depth = handle.luma_bits_per_pixel();
    let chroma = if has_alpha {
        RgbChroma::Rgba
    } else {
        RgbChroma::Rgb
    };
    // libheif applies the container's rotation and mirror properties here.
    let image = lib
        .decode(&handle, ColorSpace::Rgb(chroma), None)
        .map_err(heif)?;
    let planes = image.planes();
    let bits = planes
        .interleaved
        .as_ref()
        .map_or(0, |plane| plane.bits_per_pixel);
    let Some(plane) = planes.interleaved.filter(|plane| plane.bits_per_pixel == 8) else {
        return Err(PhotoError::UnexpectedPlanes {
            path: path.to_path_buf(),
            bits,
        });
    };
    let channels = if has_alpha { 4 } else { 3 };
    let (width, height) = (plane.width, plane.height);
    let mut rgba8 = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height as usize {
        let start = row * plane.stride;
        let row = &plane.data[start..start + width as usize * channels];
        for px in row.chunks_exact(channels) {
            rgba8.extend_from_slice(&px[..3]);
            rgba8.push(if has_alpha { px[3] } else { 255 });
        }
    }

    let source = match handle.color_profile_raw() {
        Some(profile) => slate_color::icc::source_space(&profile.data),
        None => match handle.color_profile_nclx() {
            Some(nclx) => nclx_space(nclx.color_primaries()),
            None => {
                log::info!(
                    "{} carries no colour profile; treating it as sRGB",
                    path.display()
                );
                SourceSpace::Srgb
            }
        },
    };

    let mut photo = Photo {
        width,
        height,
        rgba8,
        source,
        bit_depth,
        has_alpha,
    };
    // The container transforms are the authority on orientation; libheif
    // has applied them. An Exif orientation is honoured only when the
    // container did not turn the image, so a file that carries both is
    // not rotated twice.
    let container_turned = handle.width() != handle.ispe_width() as u32
        || handle.height() != handle.ispe_height() as u32;
    let exif = exif_orientation(&handle);
    if let Some(orientation) = exif.filter(|o| *o != 1 && !container_turned)
        && let Some(orientation) = Orientation::from_exif(orientation)
    {
        let mut image = DynamicImage::ImageRgba8(
            RgbaImage::from_raw(photo.width, photo.height, photo.rgba8)
                .expect("width times height times four bytes"),
        );
        image.apply_orientation(orientation);
        let rgba = image.into_rgba8();
        photo.width = rgba.width();
        photo.height = rgba.height();
        photo.rgba8 = rgba.into_raw();
    }
    log::info!(
        "opened {} as {}x{} {:?} through libheif, {bit_depth} bit, alpha {has_alpha}, \
         exif orientation {exif:?}, container turned {container_turned}",
        path.display(),
        photo.width,
        photo.height,
        photo.source
    );
    Ok(photo)
}

fn nclx_space(primaries: ColorPrimaries) -> SourceSpace {
    match primaries {
        ColorPrimaries::SMPTE_EG_432_1 => {
            log::info!("nclx names Display P3 primaries");
            SourceSpace::DisplayP3
        }
        ColorPrimaries::ITU_R_BT_2020_2_and_2100_0 => {
            log::info!("nclx names Rec.2020 primaries");
            SourceSpace::Icc {
                to_rec2020: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            }
        }
        other => {
            log::info!("nclx names {other:?} primaries; treating the photo as sRGB");
            SourceSpace::Srgb
        }
    }
}

/// The Exif orientation tag of the primary image, when the file has one.
fn exif_orientation(handle: &ImageHandle) -> Option<u8> {
    handle
        .all_metadata()
        .iter()
        .filter(|item| &item.item_type.0 == b"Exif")
        .find_map(|item| exif_orientation_from_block(&item.raw_data))
}

/// Reads tag 0x0112 from an Exif block as HEIF stores it: a four byte
/// big-endian offset to the TIFF header, then the TIFF structure.
fn exif_orientation_from_block(block: &[u8]) -> Option<u8> {
    let offset = u32::from_be_bytes(block.get(0..4)?.try_into().ok()?) as usize;
    let tiff = block.get(4 + offset..)?;
    let little = match tiff.get(0..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let u16_at = |at: usize| -> Option<u16> {
        let bytes: [u8; 2] = tiff.get(at..at + 2)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let bytes: [u8; 4] = tiff.get(at..at + 4)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };
    if u16_at(2)? != 42 {
        return None;
    }
    let ifd = u32_at(4)? as usize;
    let entries = u16_at(ifd)? as usize;
    (0..entries).find_map(|i| {
        let entry = ifd + 2 + i * 12;
        if u16_at(entry)? == 0x0112 {
            u16_at(entry + 8).map(|value| value as u8)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::exif_orientation_from_block;

    /// A minimal Exif block: offset 0, big-endian TIFF, one IFD entry that
    /// sets the orientation to 6.
    fn block(orientation: u16) -> Vec<u8> {
        let mut b = vec![0, 0, 0, 0];
        b.extend_from_slice(b"MM");
        b.extend_from_slice(&42u16.to_be_bytes());
        b.extend_from_slice(&8u32.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&0x0112u16.to_be_bytes());
        b.extend_from_slice(&3u16.to_be_bytes());
        b.extend_from_slice(&1u32.to_be_bytes());
        b.extend_from_slice(&orientation.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        b
    }

    #[test]
    fn reads_the_orientation_tag() {
        assert_eq!(exif_orientation_from_block(&block(6)), Some(6));
        assert_eq!(exif_orientation_from_block(&block(1)), Some(1));
    }

    #[test]
    fn a_short_block_is_none() {
        assert_eq!(exif_orientation_from_block(&[0, 0]), None);
        assert_eq!(exif_orientation_from_block(b"\0\0\0\0XX"), None);
    }
}
