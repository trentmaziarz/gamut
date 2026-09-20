//! Puts the Gamut icon on gamut-app.exe as a Windows resource.
//!
//! How the files in assets/ are made, by hand, when the mark changes:
//! 1. Each gamut-icon-<size>.svg is the drawing only. Strip the metadata
//!    element, any unused namespace and every comment from the source
//!    drawing; each file stays under 1 KB.
//! 2. Render each drawing on a transparent page with a headless browser
//!    (`msedge --headless --screenshot=<png> --window-size=256,256
//!    --default-background-color=00000000 --hide-scrollbars <page>`) and
//!    crop to the size: 16 and 24 from the 16 drawing, 32 and 48 from the
//!    32 drawing, 64, 128 and 256 from the 64 drawing.
//! 3. Pack the seven PNGs into gamut.ico (Pillow: `Image.save` with
//!    format "ICO", `sizes` and `append_images`) and keep the 256 PNG as
//!    gamut-256.png, which the window loads.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/gamut.ico");
    #[cfg(windows)]
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/gamut.ico");
        resource.compile().expect("compile the icon resource");
    }
}
