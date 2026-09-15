//! The `--export` path: render the crop at full resolution without a
//! window, resample it to the preset and write the JPEG.

use std::path::{Path, PathBuf};

use slate_core::{Crop, CropRect, ExportPreset};
use slate_gpu::Develop;
use slate_media::export::write_jpeg;

use crate::headless::{HeadlessError, Prepared};

/// Opens `photo`, applies its sidecar (or the one at `edit`), and writes
/// the crop as a JPEG of the preset's size to `out`.
pub fn write(
    photo: &Path,
    edit: Option<&Path>,
    preset: ExportPreset,
    out: &Path,
) -> Result<(), HeadlessError> {
    let prepared = Prepared::open(photo, edit)?;
    let crop = crop_for_preset(
        prepared.crop(),
        preset,
        prepared.photo.width,
        prepared.photo.height,
    );
    let gpu = &prepared.gpu;
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&prepared.photo);
    let pixels = develop
        .render_export(&prepared.sidecar.edit, crop, preset)
        .expect("the source was set");
    let (width, height) = preset.size();
    write_jpeg(&pixels, width, height, out).map_err(HeadlessError::Jpeg)
}

/// The rectangle to export for a preset: the session's crop when it has
/// the preset's aspect, else a centred crop of that aspect.
pub fn crop_for_preset(crop: Crop, preset: ExportPreset, width: u32, height: u32) -> CropRect {
    if crop.aspect == preset.aspect() {
        crop.rect
    } else {
        CropRect::fitted(preset.aspect(), width, height)
    }
}

/// The file name the export dialog proposes: the photo's stem, the preset
/// and `.jpg`.
pub fn proposed_name(photo: &Path, preset: ExportPreset) -> PathBuf {
    let stem = photo
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "photo".to_string());
    PathBuf::from(format!("{stem}_{}.jpg", preset.name()))
}

#[cfg(test)]
mod tests {
    use super::{crop_for_preset, proposed_name};
    use slate_core::{Crop, CropAspect, CropRect, ExportPreset};
    use std::path::Path;

    #[test]
    fn a_matching_crop_is_kept_and_another_aspect_is_fitted() {
        let crop = Crop {
            aspect: CropAspect::Square,
            rect: CropRect {
                x: 0.1,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
        };
        assert_eq!(
            crop_for_preset(crop, ExportPreset::Square, 2000, 1000),
            crop.rect
        );
        let fitted = crop_for_preset(crop, ExportPreset::Story9x16, 2000, 1000);
        assert_eq!(fitted, CropRect::fitted(CropAspect::Story9x16, 2000, 1000));
    }

    #[test]
    fn the_proposed_name_carries_the_preset() {
        assert_eq!(
            proposed_name(
                Path::new("C:/photos/IMG_0001.HEIC"),
                ExportPreset::Story9x16
            ),
            Path::new("IMG_0001_9x16.jpg")
        );
    }
}
