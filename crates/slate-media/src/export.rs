//! Writing the exported picture: an sRGB JPEG at quality 85.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use image::ExtendedColorType;
use image::codecs::jpeg::JpegEncoder;
use thiserror::Error;

/// The JPEG quality every export uses.
pub const QUALITY: u8 = 85;

/// Why a JPEG could not be written.
#[derive(Debug, Error)]
pub enum ExportError {
    #[error("could not create {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("could not encode {path}: {source}")]
    Encode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
}

/// Writes `rgba` (width times height times four bytes, sRGB encoded) to
/// `path` as a JPEG at [`QUALITY`]. The alpha channel is dropped.
pub fn write_jpeg(rgba: &[u8], width: u32, height: u32, path: &Path) -> Result<(), ExportError> {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "rgba is width times height times four bytes"
    );
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let file = File::create(path).map_err(|source| ExportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut encoder = JpegEncoder::new_with_quality(BufWriter::new(file), QUALITY);
    encoder
        .encode(&rgb, width, height, ExtendedColorType::Rgb8)
        .map_err(|source| ExportError::Encode {
            path: path.to_path_buf(),
            source,
        })?;
    log::info!("wrote {} at {width}x{height}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_jpeg;

    #[test]
    fn a_written_jpeg_opens_at_its_size() {
        let dir = std::env::temp_dir().join("slate-export-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("tiny.jpg");
        let (w, h) = (24, 16);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 7 % 256) as u8, 128, 30, 255])
            .collect();
        write_jpeg(&rgba, w, h, &path).expect("write");
        assert_eq!(image::image_dimensions(&path).expect("dimensions"), (w, h));
    }
}
