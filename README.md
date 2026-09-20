# Gamut

Gamut is being built as a free, open source editor for the photos and video you
shoot on your phone. The goal is to cover what people actually use in Lightroom
and DaVinci Resolve to post a 4:5 still and a 9:16 reel on Instagram, without a
subscription. The photo editor and the video editor are the same program. Gamut
runs on Windows only and is written in Rust.

Gamut is pre-release, milestone 2: a phone photo or clip opens, the Basic
sliders develop it, and it exports as a JPEG or a Reel for Instagram. With no
photo open the window shows the test image, a red-to-blue gradient under a
checkerboard. A milestone is a tagged release that adds something a user can do,
and milestone 2 is the third of the nine, numbered 0 to 8, that make version
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
