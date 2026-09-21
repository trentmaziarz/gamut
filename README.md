# Gamut

Gamut is being built as a free, open source editor for the photos and video you
shoot on your phone. The goal is to cover what people actually use in Lightroom
and DaVinci Resolve to post a 4:5 still and a 9:16 reel on Instagram, without a
subscription. The photo editor and the video editor are the same program. Gamut
runs on Windows only and is written in Rust.

Gamut is pre-release, now at milestone 3. A phone photo or clip opens, develops
on the Basic sliders and exports as a JPEG or a Reel for Instagram. Milestone 3
adds a tone curve for each colour channel, colour wheels for shadows, midtones
and highlights, and texture, clarity and dehaze sliders. A colour mixer moves
hue, saturation and luminance in eight hue ranges. An adjustment can also go to
part of the picture through a mask. You paint one with a brush, drag out a
linear or radial gradient, or pick a range of brightness or colour. Masks
combine by add, subtract and intersect. Refine edges tidies a loosely drawn
mask, moving its edge onto the nearest place where the colour changes sharply,
such as a roof against the sky. A stroke run over a row of roofs and into the
sky let go of the sky and kept the roofs. The look and the masks develop every
frame of a video clip as they do a photo. Save a look as a preset file and apply
it to another photo. One photo can carry any number of named versions of its
edit, say a colour one and a black and white one, picked by name. Gamut never
changes the photo file, and every edit lives in the small edit file it writes
beside the photo.

With no photo open the window shows the test image, a red-to-blue gradient under
a checkerboard. A milestone is a tagged release that adds something a user can
do, and milestone 3 is the fourth of the nine, numbered 0 to 8, that make version
0.1. The design document at docs/design.md in this repository lists all nine.
Bugs and questions go to the issues page at github.com/trentmaziarz/gamut.

## Build from source

1. Clone the repository with `git clone https://github.com/trentmaziarz/gamut`.
2. Install Rust stable through rustup.
3. Install Visual Studio Build Tools 2022 with the "Desktop development with C++"
   workload, which supplies the MSVC linker and the Windows SDK.
4. Install vcpkg in a folder, run `vcpkg install libheif[core]:x64-windows`, set
   VCPKG_ROOT to that folder, set VCPKGRS_DYNAMIC to 1, and add its
   `installed\x64-windows\bin` folder to PATH.
5. Unzip the FFmpeg 9 LGPL shared build from BtbN's FFmpeg-Builds releases into a
   folder, set FFMPEG_DIR to that folder and add its `bin` folder to PATH. Put
   the libclang Python wheel's libclang.dll in a folder and set LIBCLANG_PATH to
   that folder.
6. `cargo run -p gamut-app` opens the window.
7. `cargo test --workspace` runs the tests.

`cargo run -p gamut-app -- --screenshot out.png` renders the test image to a PNG,
which is how scripts and the build see the picture without opening a window.

## License

Licensed under either of Apache License 2.0 or MIT license at your option. The
two files are LICENSE-APACHE and LICENSE-MIT.
