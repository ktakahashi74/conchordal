// Entry point: launches the egui/eframe app and spawns worker threads.
mod app;
mod audio;
mod config;
mod core;
mod life;
mod synth;
mod ui;

use clap::Parser;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::life::scenario::Scenario;
use crate::life::scripting::ScriptHost;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true)]
    play: bool,

    /// Write audio to wav file
    #[arg(long)]
    wav: Option<String>,

    /// Scenario path (.json5/.rhai)
    #[arg(value_name = "SCENARIO_PATH")]
    scenario_path: String,

    /// Serialize scenario to JSON5 and exit
    #[arg(long, default_value_t = false)]
    serialize_json5: bool,
}

fn load_scenario_from_path(path: &str) -> Result<Scenario, String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rhai" => ScriptHost::load_script(path)
            .map_err(|e| format!("Failed to run scenario script {path}: {e}")),
        "json" | "json5" => {
            let contents = std::fs::read_to_string(path)
                .map_err(|err| format!("Failed to read {path}: {err}"))?;
            json5::from_str::<Scenario>(&contents)
                .map_err(|e| format!("Failed to parse scenario file {path}: {e}"))
        }
        _ => Err(format!("Unsupported scenario extension for {path}")),
    }
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();

    if args.serialize_json5 {
        let scenario = load_scenario_from_path(&args.scenario_path).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        let serialized = json5::to_string(&scenario).unwrap_or_else(|e| {
            eprintln!("Failed to serialize scenario: {e}");
            std::process::exit(1);
        });
        println!("{serialized}");
        return Ok(());
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_ctrlc = stop_flag.clone();

    ctrlc::set_handler(move || {
        stop_flag_for_ctrlc.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 1200.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Conchordal",
        native_options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, args, stop_flag.clone())))),
    )
}
