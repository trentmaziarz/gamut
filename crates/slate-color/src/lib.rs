//! slate-color is the color science on the CPU. It holds matrices between
//! sRGB, Rec.709, Rec.2020, ProPhoto and XYZ, and transfer functions for
//! sRGB, Rec.709, PQ and HLG. It holds Bradford chromatic adaptation for white
//! balance, the BT.2390 tone-mapping curve, .cube LUT parsing and ICC profiles
//! through moxcms. It also holds ACEScct, the log encoding slate runs its
//! curves and wheels in. Every operator in this crate is the reference its
//! GPU shader is tested against.

pub mod icc;

pub use icc::SourceSpace;
