//! Opens every M1 fixture and checks its size, orientation and channels.
//! The fixtures come from the fixtures-m1 release; see `fixtures::require`.

use image::{DynamicImage, ImageDecoder, ImageReader};
use slate_media::{SourceSpace, fixtures, open_photo};

fn open(name: &str) -> slate_media::Photo {
    let path = fixtures::require(name);
    open_photo(&path).unwrap_or_else(|error| panic!("open {name}: {error}"))
}

/// The top-left pixel the image crate produces once it applies the file's
/// own orientation. The photo path must agree with it.
fn reference_top_left(name: &str) -> ([u8; 4], u32, u32) {
    let path = fixtures::require(name);
    let mut decoder = ImageReader::open(&path)
        .expect("open")
        .with_guessed_format()
        .expect("guess")
        .into_decoder()
        .expect("decoder");
    let orientation = decoder.orientation().expect("orientation");
    let mut image = DynamicImage::from_decoder(decoder).expect("decode");
    image.apply_orientation(orientation);
    let rgba = image.into_rgba8();
    (rgba.get_pixel(0, 0).0, rgba.width(), rgba.height())
}

#[test]
fn example_heic_is_1280_by_854_8_bit_without_alpha() {
    let photo = open("example.heic");
    assert_eq!((photo.width, photo.height), (1280, 854));
    assert_eq!(photo.bit_depth, 8);
    assert!(!photo.has_alpha);
    assert_eq!(photo.rgba8.len(), 1280 * 854 * 4);
    assert!(photo.rgba8.as_chunks::<4>().0.iter().all(|px| px[3] == 255));
}

#[test]
fn landscape_6_comes_out_landscape() {
    let photo = open("Landscape_6.jpg");
    assert_eq!((photo.width, photo.height), (1800, 1200));
    let (pixel, width, height) = reference_top_left("Landscape_6.jpg");
    assert_eq!((width, height), (photo.width, photo.height));
    assert_eq!(photo.pixel(0, 0), pixel);
    assert_eq!(photo.source, SourceSpace::Srgb);
}

#[test]
fn portrait_8_comes_out_portrait() {
    let photo = open("Portrait_8.jpg");
    assert_eq!((photo.width, photo.height), (1200, 1800));
    let (pixel, width, height) = reference_top_left("Portrait_8.jpg");
    assert_eq!((width, height), (photo.width, photo.height));
    assert_eq!(photo.pixel(0, 0), pixel);
}

#[test]
fn alpha_512_keeps_its_alpha() {
    let photo = open("alpha_512.png");
    assert_eq!((photo.width, photo.height), (512, 512));
    assert!(photo.has_alpha);
    assert_eq!(photo.bit_depth, 8);
    assert_eq!(photo.pixel(0, 100)[3], 0);
    assert_eq!(photo.pixel(511, 100)[3], 255);
    assert!(photo.pixel(256, 100)[3] > 100 && photo.pixel(256, 100)[3] < 156);
}

#[test]
fn timing_24mp_is_6000_by_4000() {
    let photo = open("timing_24mp.jpg");
    assert_eq!((photo.width, photo.height), (6000, 4000));
    assert_eq!(photo.source, SourceSpace::Srgb);
}

#[test]
fn an_unknown_extension_is_refused() {
    let error = open_photo(std::path::Path::new("photo.tiff")).expect_err("tiff is not opened");
    assert!(error.to_string().contains("not a photo format"));
}
