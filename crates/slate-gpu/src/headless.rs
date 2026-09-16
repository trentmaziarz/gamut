//! A GPU context with no window, for tests and for the screenshot path.

use wgpu::{Adapter, Device, Instance, Queue};

/// A device and queue on an adapter that was found without a surface.
pub struct Headless {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl Headless {
    /// Creates an instance and asks for a hardware adapter first and a
    /// fallback adapter second. On Windows the fallback is DX12 WARP, which
    /// is what the windows-latest runners on GitHub offer. Returns `None`
    /// when neither exists, so a caller can skip instead of failing.
    pub fn new() -> Option<Self> {
        let instance =
            Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            request_adapter(&instance, false).or_else(|| request_adapter(&instance, true))?;
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("slate headless device"),
            required_features: crate::video::wanted_features(&adapter),
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).ok()?;
        Some(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// One line naming the adapter, for test output and logs.
    pub fn describe(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }
}

fn request_adapter(instance: &Instance, force_fallback_adapter: bool) -> Option<Adapter> {
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter,
        compatible_surface: None,
        apply_limit_buckets: false,
    };
    pollster::block_on(instance.request_adapter(&options)).ok()
}
