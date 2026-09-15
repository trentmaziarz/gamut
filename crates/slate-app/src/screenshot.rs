//! The `--screenshot` path: render without a window and write a PNG. With
//! a photo it runs the same develop graph the Viewer uses, through the
//! headless context, so CI and scripts can see the picture. Without one it
//! renders the M0 test image.

use std::path::Path;

use slate_gpu::develop::render_size_for_crop;
use slate_gpu::{Develop, Headless, Readback, TestImage};

use crate::headless::{HeadlessError, Prepared};

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
pub fn write_developed(photo: &Path, edit: Option<&Path>, out: &Path) -> Result<(), HeadlessError> {
    let prepared = Prepared::open(photo, edit)?;
    let crop = prepared.crop();
    let gpu = &prepared.gpu;
    let mut develop = Develop::new(&gpu.device, &gpu.queue);
    develop.set_source(&prepared.photo);
    let render_size = render_size_for_crop(crop.rect, SIZE);
    let view = develop
        .render(&prepared.sidecar.edit, crop.rect, render_size, SIZE)
        .expect("the source was set");
    let (width, height) = SIZE;
    let pixels = Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, view, width, height);
    save_png(out, width, height, pixels)
}

fn save_png(path: &Path, width: u32, height: u32, pixels: Vec<u8>) -> Result<(), HeadlessError> {
    let buffer = image::RgbaImage::from_raw(width, height, pixels)
        .expect("readback returns width times height times four bytes");
    buffer.save(path).map_err(HeadlessError::Image)
}
