//! Renders the test image through the headless path and reads it back.
//! The test skips, and says so, when the machine has no adapter at all.

use slate_gpu::test_image::CHECKER_CELL;
use slate_gpu::{Headless, Readback, TestImage};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;
const TOLERANCE: i32 = 8;

#[test]
fn test_image_reads_back_as_a_gradient_under_a_checker() {
    let Some(gpu) = Headless::new() else {
        println!("no adapter, skipped");
        return;
    };
    println!("adapter: {}", gpu.describe());

    let image = TestImage::new(&gpu.device, WIDTH, HEIGHT);
    image.render(&gpu.device, &gpu.queue);
    let pixels =
        Readback::new(&gpu.device).read(&gpu.device, &gpu.queue, image.view(), WIDTH, HEIGHT);
    assert_eq!(pixels.len(), (WIDTH * HEIGHT * 4) as usize);

    let pixel = |x: u32, y: u32| {
        let at = ((y * WIDTH + x) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2]]
    };

    // The checker starts with a bright cell at the top left. The bright
    // cell on the right edge sits in row 0 or row 32 depending on how
    // many cells fit across.
    let right_column_cell = (WIDTH - 1) / CHECKER_CELL;
    let right_bright_row = if right_column_cell.is_multiple_of(2) {
        0
    } else {
        CHECKER_CELL
    };

    let left = pixel(0, 0);
    let right = pixel(WIDTH - 1, right_bright_row);
    assert_close(left, [255, 0, 0], "left edge is red");
    assert_close(right, [0, 0, 255], "right edge is blue");

    // Two cells that touch along a vertical edge differ in brightness.
    let bright = pixel(0, 0);
    let dim = pixel(0, CHECKER_CELL);
    let gap = i32::from(bright[0]) - i32::from(dim[0]);
    assert!(
        gap > TOLERANCE,
        "adjacent checker cells differ: bright {bright:?}, dim {dim:?}"
    );
}

fn assert_close(actual: [u8; 3], expected: [u8; 3], what: &str) {
    for (a, e) in actual.iter().zip(expected) {
        let gap = (i32::from(*a) - i32::from(e)).abs();
        assert!(
            gap <= TOLERANCE,
            "{what}: got {actual:?}, wanted {expected:?} within {TOLERANCE}"
        );
    }
}
