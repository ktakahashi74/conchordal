#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

// Entry point: launches the egui/eframe app.
// Worker threads (including the harmonicity analysis worker) are spawned from `src/app.rs` and
// receive only the minimal data they need (fs/Log2Space/HarmonicityKernel), not a full Landscape.
mod app;
mod audio;
mod cli;
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
use tracing_subscriber::EnvFilter;

use crate::cli::Args;
use crate::config::AppConfig;

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

    let ext = Path::new(&args.scenario_path)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "rhai" {
        eprintln!("Scenario must be a .rhai script: {}", args.scenario_path);
        std::process::exit(1);
    }

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_ctrlc = stop_flag.clone();

    ctrlc::set_handler(move || {
        stop_flag_for_ctrlc.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    if args.compile_only {
        app::run_compile_only(args, config);
        return Ok(());
    }

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
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 1020.0]),
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
