use std::path::PathBuf;
use std::process::ExitCode;

use slate_app::{SlateApp, WINDOW_TITLE, native_options, screenshot};

const USAGE: &str = "usage: slate-app [<photo>]
       slate-app [--open <photo>] --screenshot <out.png> [--edit <sidecar.json>]";

#[derive(Debug, PartialEq)]
enum Command {
    Window {
        photo: Option<PathBuf>,
    },
    Screenshot {
        photo: Option<PathBuf>,
        edit: Option<PathBuf>,
        out: PathBuf,
    },
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, &'static str> {
    let mut args = args.peekable();
    let mut photo = None;
    let mut edit = None;
    let mut screenshot = None;
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
            flag if flag.starts_with("--") => return Err(USAGE),
            path if photo.is_none() => photo = Some(PathBuf::from(path)),
            _ => return Err(USAGE),
        }
    }
    match (screenshot, edit) {
        (Some(out), edit) => Ok(Command::Screenshot { photo, edit, out }),
        (None, None) => Ok(Command::Window { photo }),
        (None, Some(_)) => Err(USAGE),
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
            photo: Some(photo),
            edit,
            out,
        } => screenshot::write_developed(&photo, edit.as_deref(), &out)
            .map_err(|error| error.to_string()),
        Command::Screenshot {
            photo: None, out, ..
        } => screenshot::write(&out).map_err(|error| error.to_string()),
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
            })
        );
        assert_eq!(
            parsed(&["--screenshot", "o.png"]),
            Ok(Command::Screenshot {
                photo: None,
                edit: None,
                out: PathBuf::from("o.png"),
            })
        );
    }

    #[test]
    fn bad_arguments_print_the_usage() {
        assert!(parsed(&["--screenshot"]).is_err());
        assert!(parsed(&["--edit", "e.json"]).is_err());
        assert!(parsed(&["a.jpg", "b.jpg"]).is_err());
        assert!(parsed(&["--nope"]).is_err());
    }
}
