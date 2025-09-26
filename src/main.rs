// Entry point: launches the egui/eframe app and spawns worker threads.
mod app;
mod config;
mod core;
mod life;
mod synth;
mod ui;
mod audio;

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true)]
    play: bool,

    /// Write audio to wav file
    #[arg(long)]
    wav: Option<String>,
}

fn main() -> eframe::Result<()> {
    let args = Args::parse();

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_for_ctrlc = stop_flag.clone();

    ctrlc::set_handler(move || {
        stop_flag_for_ctrlc.store(true, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");
    
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Concord (skeleton)",
        native_options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, args, stop_flag.clone())))),
    )
}
