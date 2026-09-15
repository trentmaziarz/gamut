//! slate-color is the color science on the CPU. It holds matrices between
//! sRGB, Rec.709, Rec.2020, ProPhoto and XYZ, and transfer functions for
//! sRGB, Rec.709, PQ and HLG. It holds Bradford chromatic adaptation for white
//! balance, the BT.2390 tone-mapping curve, .cube LUT parsing and ICC profiles
//! through moxcms. It also holds ACEScct, the log encoding slate runs its
//! curves and wheels in. Every operator in this crate is the reference its
//! GPU shader is tested against.
//!
//! In M1: [`matrices`], [`transfer`], [`bradford`], [`daylight`], the six
//! Basic operators in [`basic`], and the ICC classifier in [`icc`].

pub mod basic;
pub mod bradford;
pub mod daylight;
pub mod icc;
pub mod matrices;
pub mod transfer;

pub use icc::SourceSpace;
pub use matrices::Mat3;
