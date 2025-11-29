// Entry point: launches the egui/eframe app and spawns worker threads.
mod app;
mod audio;
mod config;
mod core;
mod life;
mod synth;
mod ui;

use clap::Parser;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true)]
    play: bool,

    /// Write audio to wav file
    #[arg(long)]
    wav: Option<String>,

    /// JSON5 scenario path (required).
    #[arg(value_name = "SCENARIO_PATH")]
    scenario_path: String,
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();

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
