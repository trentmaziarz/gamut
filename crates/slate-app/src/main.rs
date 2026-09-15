use std::process::ExitCode;

use slate_app::{SlateApp, WINDOW_TITLE, native_options};

fn main() -> ExitCode {
    env_logger::init();
    let result = eframe::run_native(
        WINDOW_TITLE,
        native_options(),
        Box::new(|cc| Ok(Box::new(SlateApp::new(cc)?))),
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("slate: {error}");
            ExitCode::FAILURE
        }
    }
}
