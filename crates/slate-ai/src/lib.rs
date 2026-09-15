//! slate-ai is ONNX Runtime through the ort crate with the DirectML execution
//! provider. DirectML needs only a DX12 GPU and ships as two DLLs beside the
//! exe, with no admin install. A U2-Net or MobileSAM model gives subject masks
//! and whisper.cpp gives captions, each downloaded on first use. The crate
//! sits behind a feature flag, so the app builds without it.

#[cfg(test)]
mod tests {
    #[test]
    fn crate_links() {
        // The crate compiles and its test harness runs. Real tests arrive
        // with the milestone that fills the crate.
    }
}
