#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

// Entry point: launches the egui/eframe app.
// Worker threads (including the harmonicity analysis worker) are spawned from `src/app.rs` and
// receive only the minimal data they need (fs/Log2Space/HarmonicityKernel), not a full Landscape.
mod app;
mod audio;
mod config;
mod core;
mod life;
mod ui;

use clap::Parser;
use std::path::Path;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tracing_subscriber::EnvFilter;

use crate::config::AppConfig;
use crate::life::scenario::Scenario;
use crate::life::scripting::ScriptHost;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true")]
    play: bool,

    /// Write audio to wav file
    #[arg(long)]
    wav: Option<String>,

    /// Scenario path (.rhai only)
    #[arg(value_name = "SCENARIO_PATH")]
    scenario_path: String,

    /// Path to config TOML
    #[arg(long, default_value = "config.toml")]
    config: String,

    /// Wait for user action to exit after playback (overrides config)
    #[arg(long, num_args = 0..=1, default_missing_value = "false")]
    wait_user_exit: Option<bool>,

    /// Wait for user action before starting playback (overrides config)
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    wait_user_start: Option<bool>,

    /// Run without GUI (headless)
    #[arg(long, default_value_t = false)]
    nogui: bool,
}

fn load_scenario_from_path(path: &str) -> Result<Scenario, String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rhai" => ScriptHost::load_script(path)
            .map_err(|e| format!("Failed to run scenario script {path}: {e:#}")),
        _ => Err(format!(
            "json/json5 is not supported. Scenario must be a .rhai script: {path}"
        )),
    }
}

fn main() -> eframe::Result<()> {
    // Initialize tracing/logging (honors RUST_LOG).
    // Use an info default when RUST_LOG is unset; no wall-clock timestamps (we log sim time instead).
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .without_time()
        .try_init();

    let args = Args::parse();
    let mut config = AppConfig::load_or_default(&args.config);
    if let Some(val) = args.wait_user_exit {
        config.playback.wait_user_exit = val;
    }
    if let Some(val) = args.wait_user_start {
        config.playback.wait_user_start = val;
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_ctrlc = stop_flag.clone();

    ctrlc::set_handler(move || {
        stop_flag_for_ctrlc.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    if args.nogui {
        if config.playback.wait_user_start {
            eprintln!("--nogui forces wait_user_start=false");
        }
        if config.playback.wait_user_exit {
            eprintln!("--nogui forces wait_user_exit=false");
        }
        config.playback.wait_user_start = false;
        config.playback.wait_user_exit = false;
        app::run_headless(args, config, stop_flag);
        return Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 850.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Conchordal",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(app::App::new(
                cc,
                args.clone(),
                config.clone(),
                stop_flag.clone(),
            )))
        }),
    )
    .map_err(|e| {
        eprintln!("Error: {:?}", e);
        e
    })
}
