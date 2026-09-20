//! The `--screenshot` path: render without a window and write a PNG. With
//! a photo it runs the same develop graph the Viewer uses, through the
//! headless context, so CI and scripts can see the picture. With a video
//! or a project and `--at`, it renders the frame at that time as the
//! 1080x1920 Reel crop. Without a file it renders the M0 test image.

use std::path::Path;

use gamut_core::{Crop, CropAspect, PhotoEdit, Project};
use gamut_gpu::develop::render_size_for_crop;
use gamut_gpu::{Develop, Headless, Readback, TestImage};
use gamut_media::VideoSource;

use crate::headless::{EditSource, HeadlessError, Prepared, mask_named};
use crate::project;

/// The video screenshot size: the 9:16 Reel.
pub const VIDEO_SIZE: (u32, u32) = (1080, 1920);

/// The screenshot size: the 4:5 feed post at 1080 wide.
pub const SIZE: (u32, u32) = (1080, 1350);

/// Renders the test image at [`SIZE`] and writes it to `path` as a PNG.
pub fn write(path: &Path) -> Result<(), HeadlessError> {
    let gpu = Headless::new().ok_or(HeadlessError::NoAdapter)?;
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
pub fn write_developed(photo: &Path, source: EditSource, out: &Path) -> Result<(), HeadlessError> {
    write_developed_showing(photo, source, None, out)
}

/// [`write_developed`] with the mask of this name shown as the red overlay,
/// for `--show-mask`. An unknown name is an error that lists the masks.
pub fn write_developed_showing(
    photo: &Path,
    source: EditSource,
    show_mask: Option<&str>,
    out: &Path,
) -> Result<(), HeadlessError> {
    let prepared = Prepared::open(photo, source)?;
    let overlay = show_mask
        .map(|name| mask_named(&prepared.sidecar, name))
        .transpose()?;
    let crop = prepared.crop();
    let gpu = &prepared.gpu;
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&prepared.photo);
    develop.set_overlay(overlay);
    let render_size = render_size_for_crop(crop.rect, SIZE);
    let view = develop
        .render(&prepared.sidecar.edit, crop.rect, render_size, SIZE)
        .expect("the source was set");
    let (width, height) = SIZE;
    let pixels = Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, view, width, height);
    save_png(out, width, height, pixels)
}

/// Opens `input`, a video or a project, decodes the frame at `at` seconds
/// of the timeline, renders the 9:16 crop at [`VIDEO_SIZE`] through the
/// video pass and the develop graph, and writes it to `out` as a PNG.
pub fn write_video_frame(input: &Path, at: f64, out: &Path) -> Result<(), HeadlessError> {
    let (media, source_time, edit, crop) = if Project::is_project_path(input) {
        let loaded = project::load(input).map_err(HeadlessError::Edit)?;
        let (index, offset) =
            loaded.project.track.clip_at(at).ok_or_else(|| {
                HeadlessError::Edit(format!("{at} s is past the end of the track"))
            })?;
        let clip = loaded.project.track.clips[index];
        let media = loaded
            .project
            .media_path(&loaded.dir, clip.media)
            .ok_or_else(|| HeadlessError::Edit(format!("clip {index} names no media")))?;
        let crop =
            (loaded.project.crop.rect != gamut_core::CropRect::FULL).then_some(loaded.project.crop);
        (media, clip.source_in + offset, loaded.project.edit, crop)
    } else {
        (input.to_path_buf(), at, PhotoEdit::default(), None)
    };
    let gpu = Headless::new().ok_or(HeadlessError::NoAdapter)?;
    log::info!("adapter: {}", gpu.describe());
    let mut source = VideoSource::open(&media).map_err(HeadlessError::Video)?;
    log::info!("decoder: {:?}", source.decoder_kind);
    source.seek(source_time).map_err(HeadlessError::Video)?;
    let frame = source
        .next_frame()
        .map_err(HeadlessError::Video)?
        .ok_or_else(|| HeadlessError::Edit(format!("no frame at {source_time} s")))?;
    let crop =
        crop.unwrap_or_else(|| Crop::fitted(CropAspect::Story9x16, source.width, source.height));
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_video_frame(&frame, source.colour, source.rotation);
    let render_size = render_size_for_crop(crop.rect, VIDEO_SIZE);
    let view = develop
        .render(&edit, crop.rect, render_size, VIDEO_SIZE)
        .expect("the source was set");
    let (width, height) = VIDEO_SIZE;
    let pixels = Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, view, width, height);
    save_png(out, width, height, pixels)
}

fn save_png(path: &Path, width: u32, height: u32, pixels: Vec<u8>) -> Result<(), HeadlessError> {
    let buffer = image::RgbaImage::from_raw(width, height, pixels)
        .expect("readback returns width times height times four bytes");
    buffer.save(path).map_err(HeadlessError::Image)
}
