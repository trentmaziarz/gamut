# slate

slate is being built as a free, open source editor for the photos and video you
shoot on your phone. The goal is to cover what people actually use in Lightroom
and DaVinci Resolve to post a 4:5 still and a 9:16 reel on Instagram, without a
subscription. The photo editor and the video editor are the same program. slate
runs on Windows only and is written in Rust.

slate is pre-release, milestone 0: the window opens and draws a test image,
nothing edits yet. The test image is a red-to-blue gradient under a
checkerboard. A milestone is a tagged release that adds something a user can do,
and milestone 0 is the first of the nine that make version 0.1. The design
document at docs/design.md in this repository lists all nine. Bugs and questions
go to the issues page at github.com/trentmaziarz/slate.

## Build from source

1. Clone the repository with `git clone https://github.com/trentmaziarz/slate`.
2. Install Rust stable through rustup.
3. Install Visual Studio Build Tools 2022 with the "Desktop development with C++"
   workload, which supplies the MSVC linker and the Windows SDK.
4. Run `cargo run -p slate-app` to open the window.
5. Run `cargo test --workspace` to run the tests.

`cargo run -p slate-app -- --screenshot out.png` renders the test image to a PNG.
Scripts and the build use it to see the picture without opening a window.

## License

Licensed under either of Apache License 2.0 or MIT license at your option. The
two files are LICENSE-APACHE and LICENSE-MIT.
