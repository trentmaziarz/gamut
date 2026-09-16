//! The hardware decode device: the one unsafe module of slate. ffmpeg's safe
//! wrappers expose no hwaccel, so this module creates the CUDA device
//! context with `av_hwdevice_ctx_create`, attaches it to a decoder before
//! the decoder opens, installs the `get_format` callback that picks the
//! CUDA pixel format when the decoder offers it, and moves each decoded
//! CUDA frame into system memory with `av_hwframe_transfer_data`. It also
//! holds the one raw read of a stream's coded side data, because FFmpeg 9
//! moved the display matrix there and the safe wrapper has no accessor yet.
//! Nothing else in the workspace touches a raw ffmpeg pointer.
//!
//! Every unsafe block below states the invariant it relies on.

use ffmpeg::ffi;
use ffmpeg::{Error, codec, frame};
use ffmpeg_the_third as ffmpeg;
use std::ptr;

/// A reference to a hardware device context. Dropping it releases the
/// reference; the decoder that was attached holds its own.
pub struct HwDevice {
    device: *mut ffi::AVBufferRef,
}

// SAFETY: an AVBufferRef to a device context is reference counted with
// atomic operations inside ffmpeg, and the CUDA device context is documented
// as usable from any thread. The pointer is only ever read to take a new
// reference or released in Drop.
unsafe impl Send for HwDevice {}

impl HwDevice {
    /// Creates the CUDA device context. `None` when no CUDA device answers,
    /// which is the case on a machine without an NVIDIA GPU and on CI.
    pub fn cuda() -> Option<Self> {
        let mut device: *mut ffi::AVBufferRef = ptr::null_mut();
        // SAFETY: `device` is a valid out pointer to a null AVBufferRef
        // pointer; the device name is null, which asks for the default CUDA
        // device; the options are null. On success ffmpeg writes a new
        // reference that this struct owns and releases in Drop.
        let result = unsafe {
            ffi::av_hwdevice_ctx_create(
                &mut device,
                ffi::AVHWDeviceType::CUDA,
                ptr::null(),
                ptr::null_mut(),
                0,
            )
        };
        if result < 0 || device.is_null() {
            log::info!(
                "no CUDA device context: {}; decoding in software",
                Error::from(result)
            );
            return None;
        }
        Some(Self { device })
    }

    /// Attaches this device to a decoder context that has not been opened
    /// yet, and installs the format callback that chooses CUDA frames.
    pub fn attach(&self, context: &mut codec::Context) {
        // SAFETY: `context` wraps a live AVCodecContext that has not been
        // opened, so its fields may be written. `av_buffer_ref` takes a new
        // reference to the device that the codec context owns from here on
        // and releases when it is freed. The callback has the exact
        // signature avcodec.h declares for `get_format`.
        unsafe {
            let raw = context.as_mut_ptr();
            (*raw).hw_device_ctx = ffi::av_buffer_ref(self.device);
            (*raw).get_format = Some(pick_cuda);
        }
    }
}

impl Drop for HwDevice {
    fn drop(&mut self) {
        // SAFETY: `device` is the reference `av_hwdevice_ctx_create` gave
        // this struct and nothing else releases it. `av_buffer_unref`
        // accepts a pointer to the reference and nulls it.
        unsafe { ffi::av_buffer_unref(&mut self.device) }
    }
}

/// The `get_format` callback: walks the formats the decoder offers and
/// returns CUDA when it is there, else the first offered format so that
/// the decoder falls back to software output instead of failing.
unsafe extern "C" fn pick_cuda(
    _context: *mut ffi::AVCodecContext,
    formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    // SAFETY: ffmpeg passes a list of pixel formats terminated by
    // AV_PIX_FMT_NONE; the loop reads each entry until it meets that
    // terminator and never past it.
    unsafe {
        let mut at = formats;
        while *at != ffi::AVPixelFormat::NONE {
            if *at == ffi::AVPixelFormat::CUDA {
                return ffi::AVPixelFormat::CUDA;
            }
            at = at.add(1);
        }
        *formats
    }
}

/// Whether a decoded frame lives in CUDA memory.
pub fn is_hardware_frame(frame: &frame::Video) -> bool {
    frame.format() == ffmpeg::format::Pixel::CUDA
}

/// Copies a CUDA frame into `system`, a frame in system memory. When
/// `system` already holds a buffer of the right format and size it is
/// reused; otherwise ffmpeg allocates one. The frame properties (timestamps,
/// colour tags) are copied too.
pub fn transfer(hardware: &frame::Video, system: &mut frame::Video) -> Result<(), Error> {
    // SAFETY: both frames wrap live AVFrames owned by the caller for the
    // duration of the call; `hardware` holds a CUDA frame from a decoder
    // attached to a device from this module, so its hw_frames_ctx is set,
    // which is what av_hwframe_transfer_data requires of its source.
    let result =
        unsafe { ffi::av_hwframe_transfer_data(system.as_mut_ptr(), hardware.as_ptr(), 0) };
    if result < 0 {
        return Err(Error::from(result));
    }
    Ok(())
}

/// The display matrix a stream carries in its coded side data, as the nine
/// 16.16 fixed point values, or `None` when it has none.
pub fn display_matrix(parameters: &codec::ParametersRef<'_>) -> Option<[i32; 9]> {
    // SAFETY: `parameters` borrows a live AVCodecParameters for its
    // lifetime; coded_side_data holds nb_coded_side_data entries, which is
    // what av_packet_side_data_get is given; the returned entry, when not
    // null, points at at least `size` bytes of data.
    unsafe {
        let raw = parameters.as_ptr();
        let entry = ffi::av_packet_side_data_get(
            (*raw).coded_side_data,
            (*raw).nb_coded_side_data,
            ffi::AVPacketSideDataType::DISPLAYMATRIX,
        );
        if entry.is_null() || (*entry).size < 36 || (*entry).data.is_null() {
            return None;
        }
        let bytes = std::slice::from_raw_parts((*entry).data, 36);
        let mut matrix = [0i32; 9];
        for (i, value) in matrix.iter_mut().enumerate() {
            let at = i * 4;
            *value = i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        }
        Some(matrix)
    }
}

/// Whether the CUDA device context can be created on this machine. Tests
/// that need NVDEC skip when this is false.
pub fn nvdec_available() -> bool {
    crate::init_ffmpeg();
    HwDevice::cuda().is_some()
}
