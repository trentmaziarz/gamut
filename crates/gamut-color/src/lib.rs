//! gamut-color is the color science on the CPU. It holds matrices between
//! sRGB, Rec.709, Rec.2020, ProPhoto and XYZ, and transfer functions for
//! sRGB, Rec.709, PQ and HLG. It holds Bradford chromatic adaptation for white
//! balance, the BT.2390 tone-mapping curve, .cube LUT parsing and ICC profiles
//! through moxcms. It also holds ACEScct, the log encoding Gamut runs its
//! curves and wheels in. Every operator in this crate is the reference its
//! GPU shader is tested against.
//!
//! In M1: [`matrices`], [`transfer`], [`bradford`], [`daylight`], the six
//! Basic operators in [`basic`], and the ICC classifier in [`icc`]. M2 adds
//! [`video`], the twin of the YCbCr pass with the provisional HLG decode.
//! M3 adds [`acescct`] and the operators that run in it ([`curve`], [`hsl`]
//! on the hue model of [`hue`], [`wheels`]), and the local operators
//! [`local`] (texture and clarity) and [`dehaze`], the masks of [`mask`] with
//! the painted ones of [`brush`], and [`refine`], the edge-aware filter of a
//! mask's alpha.

pub mod acescct;
pub mod basic;
pub mod bradford;
pub mod brush;
pub mod curve;
pub mod daylight;
pub mod dehaze;
pub mod edge;
pub mod hsl;
pub mod hue;
pub mod icc;
pub mod local;
pub mod mask;
pub mod matrices;
pub mod refine;
pub mod transfer;
pub mod video;
pub mod wheels;

pub use icc::SourceSpace;
pub use matrices::Mat3;
