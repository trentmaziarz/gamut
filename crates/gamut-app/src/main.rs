use std::path::PathBuf;
use std::process::ExitCode;

use gamut_app::headless::EditSource;
use gamut_app::{GamutApp, WINDOW_TITLE, export, native_options, reel, screenshot};
use gamut_core::ExportPreset;

const USAGE: &str = "usage: gamut-app [<photo, video or project.gamut>]
       gamut-app [--open <photo>] --screenshot <out.png> [look] [--show-mask <name>]
       gamut-app --open <video or project.gamut> --screenshot <out.png> --at <seconds>
       gamut-app --open <photo> --export <4x5|1x1|3x4|9x16> <out.jpg> [look]
       gamut-app --export-reel <video or project.gamut> <out.mp4>
look:  [--edit <sidecar.json>] [--version-name <name>] [--preset <look.json>]
       the named version of the sidecar, then the look preset over it
--show-mask draws the named mask of the edit as a red overlay on a screenshot";

#[derive(Debug, PartialEq)]
enum Command {
    Window {
        photo: Option<PathBuf>,
    },
    Screenshot {
        photo: Option<PathBuf>,
        edit: Option<PathBuf>,
        version: Option<String>,
        look: Option<PathBuf>,
        show_mask: Option<String>,
        out: PathBuf,
        at: Option<f64>,
    },
    ExportReel {
        input: PathBuf,
        out: PathBuf,
    },
    Export {
        photo: PathBuf,
        edit: Option<PathBuf>,
        version: Option<String>,
        look: Option<PathBuf>,
        preset: ExportPreset,
        out: PathBuf,
    },
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, &'static str> {
    let mut args = args.peekable();
    let mut photo = None;
    let mut edit = None;
    let mut screenshot = None;
    let mut export = None;
    let mut at = None;
    let mut reel = None;
    let mut look = None;
    let mut version = None;
    let mut show_mask = None;
    while let Some(arg) = args.next() {
        let mut value = |slot: &mut Option<PathBuf>| match args.next() {
            Some(path) if slot.is_none() => {
                *slot = Some(PathBuf::from(path));
                Ok(())
            }
            _ => Err(USAGE),
        };
        match arg.as_str() {
            "--open" => value(&mut photo)?,
            "--screenshot" => value(&mut screenshot)?,
            "--edit" => value(&mut edit)?,
            "--preset" => value(&mut look)?,
            "--version-name" => match (args.next(), &version) {
                (Some(name), None) if !name.trim().is_empty() => version = Some(name),
                _ => return Err(USAGE),
            },
            "--show-mask" => match (args.next(), &show_mask) {
                (Some(name), None) if !name.trim().is_empty() => show_mask = Some(name),
                _ => return Err(USAGE),
            },
            "--at" => {
                let seconds = args.next().and_then(|s| s.parse::<f64>().ok());
                match (seconds, &at) {
                    (Some(seconds), None) if seconds >= 0.0 => at = Some(seconds),
                    _ => return Err(USAGE),
                }
            }
            "--export-reel" => match (args.next(), args.next(), &reel) {
                (Some(input), Some(out), None) => {
                    reel = Some((PathBuf::from(input), PathBuf::from(out)));
                }
                _ => return Err(USAGE),
            },
            "--export" => {
                let preset = args.next().and_then(|p| ExportPreset::parse(&p));
                match (preset, args.next(), &export) {
                    (Some(preset), Some(out), None) => {
                        export = Some((preset, PathBuf::from(out)));
                    }
                    _ => return Err(USAGE),
                }
            }
            flag if flag.starts_with("--") => return Err(USAGE),
            path if photo.is_none() => photo = Some(PathBuf::from(path)),
            _ => return Err(USAGE),
        }
    }
    if let Some((input, out)) = reel {
        let plain = photo.is_none() && screenshot.is_none() && export.is_none();
        let no_look = edit.is_none() && look.is_none() && version.is_none() && show_mask.is_none();
        return if plain && no_look && at.is_none() {
            Ok(Command::ExportReel { input, out })
        } else {
            Err(USAGE)
        };
    }
    // A look needs a photo to land on, and a video frame takes none.
    let has_look = edit.is_some() || look.is_some() || version.is_some();
    if (look.is_some() || version.is_some() || show_mask.is_some()) && photo.is_none() {
        return Err(USAGE);
    }
    match (screenshot, export) {
        (Some(out), None) => {
            if at.is_some() && (photo.is_none() || has_look || show_mask.is_some()) {
                return Err(USAGE);
            }
            Ok(Command::Screenshot {
                photo,
                edit,
                version,
                look,
                show_mask,
                out,
                at,
            })
        }
        // The overlay is a view of a mask: it shows on a screenshot only.
        _ if show_mask.is_some() => Err(USAGE),
        (None, Some((preset, out))) if at.is_none() => match photo {
            Some(photo) => Ok(Command::Export {
                photo,
                edit,
                version,
                look,
                preset,
                out,
            }),
            None => Err(USAGE),
        },
        (None, None) if at.is_none() && !has_look => Ok(Command::Window { photo }),
        _ => Err(USAGE),
    }
}

fn main() -> ExitCode {
    env_logger::init();
    let command = match parse(std::env::args().skip(1)) {
        Ok(command) => command,
        Err(usage) => {
            eprintln!("{usage}");
            return ExitCode::FAILURE;
        }
    };
    let result = match command {
        Command::Screenshot {
            photo: Some(input),
            out,
            at: Some(at),
            ..
        } => screenshot::write_video_frame(&input, at, &out).map_err(|error| error.to_string()),
        Command::Screenshot {
            photo: Some(photo),
            edit,
            version,
            look,
            show_mask,
            out,
            at: None,
        } => {
            let source = EditSource {
                edit: edit.as_deref(),
                version: version.as_deref(),
                look: look.as_deref(),
            };
            screenshot::write_developed_showing(&photo, source, show_mask.as_deref(), &out)
                .map_err(|error| error.to_string())
        }
        Command::Screenshot {
            photo: None, out, ..
        } => screenshot::write(&out).map_err(|error| error.to_string()),
        Command::ExportReel { input, out } => reel::export_file(&input, &out)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        Command::Export {
            photo,
            edit,
            version,
            look,
            preset,
            out,
        } => {
            let source = EditSource {
                edit: edit.as_deref(),
                version: version.as_deref(),
                look: look.as_deref(),
            };
            export::write(&photo, source, preset, &out).map_err(|error| error.to_string())
        }
        Command::Window { photo } => eframe::run_native(
            WINDOW_TITLE,
            native_options(),
            Box::new(move |cc| Ok(Box::new(GamutApp::new(cc, photo)?))),
        )
        .map_err(|error| error.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("slate: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, parse};
    use gamut_core::ExportPreset;
    use std::path::PathBuf;

    fn parsed(args: &[&str]) -> Result<Command, &'static str> {
        parse(args.iter().map(|a| a.to_string()))
    }

    #[test]
    fn show_mask_goes_with_a_photo_screenshot_and_nothing_else() {
        assert_eq!(
            parsed(&[
                "--open",
                "a.jpg",
                "--edit",
                "e.json",
                "--screenshot",
                "o.png",
                "--show-mask",
                "Sky"
            ]),
            Ok(Command::Screenshot {
                photo: Some(PathBuf::from("a.jpg")),
                edit: Some(PathBuf::from("e.json")),
                version: None,
                look: None,
                show_mask: Some("Sky".to_string()),
                out: PathBuf::from("o.png"),
                at: None,
            })
        );
        let with = |rest: &[&str]| {
            let mut args = vec!["--show-mask", "Sky"];
            args.extend_from_slice(rest);
            parsed(&args)
        };
        assert!(with(&["--open", "a.jpg", "--export", "4x5", "o.jpg"]).is_err());
        assert!(with(&["--export-reel", "p.gamut", "o.mp4"]).is_err());
        assert!(with(&["--open", "v.mp4", "--screenshot", "o.png", "--at", "1.5"]).is_err());
        assert!(with(&["--screenshot", "o.png"]).is_err(), "no photo");
        assert!(with(&["a.jpg"]).is_err(), "the window takes none");
        assert!(
            with(&[
                "--open",
                "a.jpg",
                "--screenshot",
                "o.png",
                "--show-mask",
                "B"
            ])
            .is_err()
        );
        assert!(parsed(&["--open", "a.jpg", "--screenshot", "o.png", "--show-mask"]).is_err());
        assert!(
            parsed(&[
                "--open",
                "a.jpg",
                "--screenshot",
                "o.png",
                "--show-mask",
                " "
            ])
            .is_err()
        );
    }

    #[test]
    fn no_arguments_opens_the_window() {
        assert_eq!(parsed(&[]), Ok(Command::Window { photo: None }));
    }

    #[test]
    fn a_bare_path_opens_the_photo_in_the_window() {
        assert_eq!(
            parsed(&["a.heic"]),
            Ok(Command::Window {
                photo: Some(PathBuf::from("a.heic"))
            })
        );
    }

    #[test]
    fn screenshot_with_open_and_edit() {
        assert_eq!(
            parsed(&[
                "--open",
                "a.jpg",
                "--screenshot",
                "o.png",
                "--edit",
                "e.json"
            ]),
            Ok(Command::Screenshot {
                photo: Some(PathBuf::from("a.jpg")),
                edit: Some(PathBuf::from("e.json")),
                version: None,
                look: None,
                show_mask: None,
                out: PathBuf::from("o.png"),
                at: None,
            })
        );
        assert_eq!(
            parsed(&["--screenshot", "o.png"]),
            Ok(Command::Screenshot {
                photo: None,
                edit: None,
                version: None,
                look: None,
                show_mask: None,
                out: PathBuf::from("o.png"),
                at: None,
            })
        );
    }

    #[test]
    fn a_preset_and_a_version_name_ride_with_a_screenshot_or_an_export() {
        assert_eq!(
            parsed(&[
                "--open",
                "a.jpg",
                "--edit",
                "e.json",
                "--version-name",
                "Warm one",
                "--preset",
                "look.json",
                "--screenshot",
                "o.png",
            ]),
            Ok(Command::Screenshot {
                photo: Some(PathBuf::from("a.jpg")),
                edit: Some(PathBuf::from("e.json")),
                version: Some("Warm one".to_string()),
                look: Some(PathBuf::from("look.json")),
                show_mask: None,
                out: PathBuf::from("o.png"),
                at: None,
            })
        );
        assert_eq!(
            parsed(&[
                "--open", "a.jpg", "--preset", "l.json", "--export", "4x5", "o.jpg"
            ]),
            Ok(Command::Export {
                photo: PathBuf::from("a.jpg"),
                edit: None,
                version: None,
                look: Some(PathBuf::from("l.json")),
                preset: ExportPreset::Feed4x5,
                out: PathBuf::from("o.jpg"),
            })
        );
    }

    #[test]
    fn a_preset_or_a_version_name_is_refused_where_it_cannot_apply() {
        // Not with a Reel export.
        assert!(parsed(&["--export-reel", "p.gamut", "o.mp4", "--preset", "l.json"]).is_err());
        assert!(parsed(&["--export-reel", "p.gamut", "o.mp4", "--version-name", "a"]).is_err());
        // Not on a video frame, not without a photo, not for the window.
        assert!(
            parsed(&[
                "--open",
                "a.mp4",
                "--screenshot",
                "o.png",
                "--at",
                "1",
                "--preset",
                "l.json"
            ])
            .is_err()
        );
        assert!(parsed(&["--screenshot", "o.png", "--preset", "l.json"]).is_err());
        assert!(parsed(&["a.jpg", "--preset", "l.json"]).is_err());
        // Not twice, not empty, not without a value.
        assert!(
            parsed(&[
                "--open",
                "a.jpg",
                "--screenshot",
                "o.png",
                "--preset",
                "a",
                "--preset",
                "b"
            ])
            .is_err()
        );
        assert!(
            parsed(&[
                "--open",
                "a.jpg",
                "--screenshot",
                "o.png",
                "--version-name",
                " "
            ])
            .is_err()
        );
        assert!(parsed(&["--open", "a.jpg", "--screenshot", "o.png", "--version-name"]).is_err());
    }

    #[test]
    fn a_video_screenshot_takes_a_time() {
        assert_eq!(
            parsed(&["--open", "a.mp4", "--screenshot", "o.png", "--at", "3.0"]),
            Ok(Command::Screenshot {
                photo: Some(PathBuf::from("a.mp4")),
                edit: None,
                version: None,
                look: None,
                show_mask: None,
                out: PathBuf::from("o.png"),
                at: Some(3.0),
            })
        );
        assert!(parsed(&["--screenshot", "o.png", "--at", "3.0"]).is_err());
        assert!(parsed(&["--open", "a.mp4", "--screenshot", "o.png", "--at", "x"]).is_err());
        assert!(parsed(&["--open", "a.mp4", "--at", "1"]).is_err());
    }

    #[test]
    fn a_reel_export_takes_an_input_and_an_output() {
        assert_eq!(
            parsed(&["--export-reel", "p.gamut", "o.mp4"]),
            Ok(Command::ExportReel {
                input: PathBuf::from("p.gamut"),
                out: PathBuf::from("o.mp4"),
            })
        );
        assert!(parsed(&["--export-reel", "p.gamut"]).is_err());
        assert!(parsed(&["--export-reel", "p.gamut", "o.mp4", "--screenshot", "s.png"]).is_err());
    }

    #[test]
    fn export_takes_a_preset_and_a_path() {
        assert_eq!(
            parsed(&["--open", "a.heic", "--export", "9x16", "o.jpg"]),
            Ok(Command::Export {
                photo: PathBuf::from("a.heic"),
                edit: None,
                version: None,
                look: None,
                preset: ExportPreset::Story9x16,
                out: PathBuf::from("o.jpg"),
            })
        );
        assert!(parsed(&["--export", "9x16", "o.jpg"]).is_err());
        assert!(parsed(&["--open", "a.heic", "--export", "16x9", "o.jpg"]).is_err());
        assert!(parsed(&["--open", "a.heic", "--export", "9x16"]).is_err());
    }

    #[test]
    fn bad_arguments_print_the_usage() {
        assert!(parsed(&["--screenshot"]).is_err());
        assert!(parsed(&["--edit", "e.json"]).is_err());
        assert!(parsed(&["a.jpg", "b.jpg"]).is_err());
        assert!(parsed(&["--nope"]).is_err());
        assert!(parsed(&["--screenshot", "o.png", "--export", "4x5", "o.jpg"]).is_err());
    }
}
