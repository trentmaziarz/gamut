//! The crop of a photo: one of the four Instagram aspects and where it sits.

use serde::{Deserialize, Serialize};

/// The four aspects a photo is cropped to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CropAspect {
    /// The 4:5 feed post.
    #[default]
    Feed4x5,
    /// The 1:1 square.
    Square,
    /// The 3:4 profile grid.
    Grid3x4,
    /// The 9:16 story and reel.
    Story9x16,
}

impl CropAspect {
    pub const ALL: [CropAspect; 4] = [
        CropAspect::Feed4x5,
        CropAspect::Square,
        CropAspect::Grid3x4,
        CropAspect::Story9x16,
    ];

    /// Width over height.
    pub fn ratio(self) -> f32 {
        let (w, h) = self.parts();
        w / h
    }

    /// Width and height as the small integers of the name.
    pub fn parts(self) -> (f32, f32) {
        match self {
            CropAspect::Feed4x5 => (4.0, 5.0),
            CropAspect::Square => (1.0, 1.0),
            CropAspect::Grid3x4 => (3.0, 4.0),
            CropAspect::Story9x16 => (9.0, 16.0),
        }
    }

    /// The name on the button.
    pub fn label(self) -> &'static str {
        match self {
            CropAspect::Feed4x5 => "4:5",
            CropAspect::Square => "1:1",
            CropAspect::Grid3x4 => "3:4",
            CropAspect::Story9x16 => "9:16",
        }
    }
}

/// A rectangle in the photo, in fractions of its width and height.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CropRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CropRect {
    /// The whole photo.
    pub const FULL: CropRect = CropRect {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    /// The largest centred rectangle of `aspect` inside a photo of
    /// `photo_width` by `photo_height` pixels.
    pub fn fitted(aspect: CropAspect, photo_width: u32, photo_height: u32) -> CropRect {
        let photo = photo_width as f32 / photo_height as f32;
        let wanted = aspect.ratio();
        let (width, height) = if wanted < photo {
            (wanted / photo, 1.0)
        } else {
            (1.0, photo / wanted)
        };
        CropRect {
            x: (1.0 - width) / 2.0,
            y: (1.0 - height) / 2.0,
            width,
            height,
        }
    }

    /// The rectangle moved by `dx`, `dy` and kept inside the photo.
    pub fn moved(self, dx: f32, dy: f32) -> CropRect {
        CropRect {
            x: (self.x + dx).clamp(0.0, 1.0 - self.width),
            y: (self.y + dy).clamp(0.0, 1.0 - self.height),
            ..self
        }
    }

    /// The size in pixels of this rectangle on a photo of the given size.
    pub fn pixel_size(self, photo_width: u32, photo_height: u32) -> (u32, u32) {
        (
            (self.width * photo_width as f32).round().max(1.0) as u32,
            (self.height * photo_height as f32).round().max(1.0) as u32,
        )
    }
}

impl Default for CropRect {
    fn default() -> Self {
        CropRect::FULL
    }
}

/// The crop of a photo: the aspect and the rectangle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Crop {
    pub aspect: CropAspect,
    pub rect: CropRect,
}

impl Crop {
    /// A centred crop of `aspect` on a photo of the given size.
    pub fn fitted(aspect: CropAspect, photo_width: u32, photo_height: u32) -> Crop {
        Crop {
            aspect,
            rect: CropRect::fitted(aspect, photo_width, photo_height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Crop, CropAspect, CropRect};

    #[test]
    fn a_crop_round_trips_through_json() {
        let crop = Crop {
            aspect: CropAspect::Story9x16,
            rect: CropRect {
                x: 0.1,
                y: 0.0,
                width: 0.5,
                height: 1.0,
            },
        };
        let text = serde_json::to_string(&crop).expect("serialize");
        let back: Crop = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(crop, back);
        let empty: Crop = serde_json::from_str("{}").expect("empty");
        assert_eq!(empty, Crop::default());
    }

    #[test]
    fn a_fitted_crop_is_centred_and_has_the_aspect() {
        let rect = CropRect::fitted(CropAspect::Feed4x5, 3000, 2000);
        let (w, h) = rect.pixel_size(3000, 2000);
        assert_eq!((w, h), (1600, 2000));
        assert!((rect.x - (1.0 - 1600.0 / 3000.0) / 2.0).abs() < 1e-6);
        assert_eq!(rect.y, 0.0);
        let tall = CropRect::fitted(CropAspect::Story9x16, 1200, 1800);
        assert_eq!(tall.pixel_size(1200, 1800), (1013, 1800));
        let wide = CropRect::fitted(CropAspect::Square, 1000, 4000);
        assert_eq!(wide.pixel_size(1000, 4000), (1000, 1000));
    }

    #[test]
    fn a_moved_crop_stays_inside_the_photo() {
        let rect = CropRect::fitted(CropAspect::Square, 2000, 1000);
        let moved = rect.moved(10.0, 10.0);
        assert!((moved.x - 0.5).abs() < 1e-6);
        assert_eq!(moved.y, 0.0);
        let back = rect.moved(-10.0, -10.0);
        assert_eq!((back.x, back.y), (0.0, 0.0));
    }
}
