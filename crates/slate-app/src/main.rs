use std::path::PathBuf;
use std::process::ExitCode;

use slate_app::{SlateApp, WINDOW_TITLE, native_options, screenshot};

const USAGE: &str = "usage: slate-app [--screenshot <path>]";

enum Command {
    Window,
    Screenshot(PathBuf),
}

fn parse(args: impl Iterator<Item = String>) -> Result<Command, &'static str> {
    let args: Vec<String> = args.collect();
    match args.as_slice() {
        [] => Ok(Command::Window),
        [flag, path] if flag == "--screenshot" => Ok(Command::Screenshot(PathBuf::from(path))),
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
        Command::Screenshot(path) => screenshot::write(&path).map_err(|error| error.to_string()),
        Command::Window => eframe::run_native(
            WINDOW_TITLE,
            native_options(),
            Box::new(|cc| Ok(Box::new(SlateApp::new(cc)?))),
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
