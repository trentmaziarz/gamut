use std::path::PathBuf;
use std::process::ExitCode;

use slate_app::{SlateApp, WINDOW_TITLE, export, native_options, reel, screenshot};
use slate_core::ExportPreset;

const USAGE: &str = "usage: slate-app [<photo, video or project.slate>]
       slate-app [--open <photo>] --screenshot <out.png> [--edit <sidecar.json>]
       slate-app --open <video or project.slate> --screenshot <out.png> --at <seconds>
       slate-app --open <photo> --export <4x5|1x1|3x4|9x16> <out.jpg> [--edit <sidecar.json>]
       slate-app --export-reel <video or project.slate> <out.mp4>";

#[derive(Debug, PartialEq)]
enum Command {
    Window {
        photo: Option<PathBuf>,
    },
    Screenshot {
        photo: Option<PathBuf>,
        edit: Option<PathBuf>,
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
        return match (photo, screenshot, export, edit, at) {
            (None, None, None, None, None) => Ok(Command::ExportReel { input, out }),
            _ => Err(USAGE),
        };
    }
    match (screenshot, export, edit) {
        (Some(out), None, edit) => {
            if at.is_some() && (photo.is_none() || edit.is_some()) {
                return Err(USAGE);
            }
            Ok(Command::Screenshot {
                photo,
                edit,
                out,
                at,
            })
        }
        (None, Some((preset, out)), edit) if at.is_none() => match photo {
            Some(photo) => Ok(Command::Export {
                photo,
                edit,
                preset,
                out,
            }),
            None => Err(USAGE),
        },
        (None, None, None) if at.is_none() => Ok(Command::Window { photo }),
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
            out,
            at: None,
        } => screenshot::write_developed(&photo, edit.as_deref(), &out)
            .map_err(|error| error.to_string()),
        Command::Screenshot {
            photo: None, out, ..
        } => screenshot::write(&out).map_err(|error| error.to_string()),
        Command::ExportReel { input, out } => reel::export_file(&input, &out)
            .map(|_| ())
            .map_err(|error| error.to_string()),
        Command::Export {
            photo,
            edit,
            preset,
            out,
        } => {
            export::write(&photo, edit.as_deref(), preset, &out).map_err(|error| error.to_string())
        }
        Command::Window { photo } => eframe::run_native(
            WINDOW_TITLE,
            native_options(),
            Box::new(move |cc| Ok(Box::new(SlateApp::new(cc, photo)?))),
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
    use slate_core::ExportPreset;
    use std::path::PathBuf;

    fn parsed(args: &[&str]) -> Result<Command, &'static str> {
        parse(args.iter().map(|a| a.to_string()))
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
                out: PathBuf::from("o.png"),
                at: None,
            })
        );
        assert_eq!(
            parsed(&["--screenshot", "o.png"]),
            Ok(Command::Screenshot {
                photo: None,
                edit: None,
                out: PathBuf::from("o.png"),
                at: None,
            })
        );
    }

    #[test]
    fn a_video_screenshot_takes_a_time() {
        assert_eq!(
            parsed(&["--open", "a.mp4", "--screenshot", "o.png", "--at", "3.0"]),
            Ok(Command::Screenshot {
                photo: Some(PathBuf::from("a.mp4")),
                edit: None,
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
            parsed(&["--export-reel", "p.slate", "o.mp4"]),
            Ok(Command::ExportReel {
                input: PathBuf::from("p.slate"),
                out: PathBuf::from("o.mp4"),
            })
        );
        assert!(parsed(&["--export-reel", "p.slate"]).is_err());
        assert!(parsed(&["--export-reel", "p.slate", "o.mp4", "--screenshot", "s.png"]).is_err());
    }

    #[test]
    fn export_takes_a_preset_and_a_path() {
        assert_eq!(
            parsed(&["--open", "a.heic", "--export", "9x16", "o.jpg"]),
            Ok(Command::Export {
                photo: PathBuf::from("a.heic"),
                edit: None,
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
