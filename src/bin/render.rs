//! conchordal-render: offline WAV renderer.
//! Shares the core engine with the Conchordal instrument but writes
//! audio to disk instead of playing through an audio device.
//! This binary is NOT the instrument; the instrument's air-gap policy
//! does not apply here.

use clap::Parser;
use std::sync::{Arc, atomic::AtomicBool};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "conchordal-render",
    about = "Offline WAV renderer for Conchordal scenarios"
)]
struct Args {
    /// Scenario script path (.rhai)
    #[arg(value_name = "SCENARIO")]
    scenario: String,

    /// Output WAV file path
    #[arg(short, long)]
    output: String,

    /// Path to config TOML
    #[arg(long, default_value = "config.toml")]
    config: String,
}

fn main() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_level(true)
        .without_time()
        .try_init();

    let args = Args::parse();
    let config = conchordal::config::AppConfig::load_or_default(&args.config);

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_ctrlc = stop_flag.clone();
    ctrlc::set_handler(move || {
        stop_flag_ctrlc.store(true, std::sync::atomic::Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    if let Err(err) = conchordal::app::run_render(&args.scenario, args.output, config, stop_flag) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
