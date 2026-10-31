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
    #[arg(long)]
    config: Option<String>,

    /// Override the fresh scenario seed (a script-level seed() still wins)
    #[arg(long)]
    seed: Option<u64>,

    /// Request a graceful end at this score time, followed by release fades
    #[arg(long, value_name = "SEC", value_parser = conchordal::cli::parse_duration)]
    duration: Option<f64>,

    /// Write JSONL telemetry alongside this render's WAV
    #[arg(long, value_name = "PATH")]
    report: Option<String>,

    /// Keep runtime-generated Voice IDs above this bound for matched scenario comparisons
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u64).range(..u64::MAX))]
    reserve_runtime_ids_through: u64,
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
    let config = match conchordal::config::AppConfig::load_or_default(args.config.as_deref()) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(1);
        }
    };

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_ctrlc = stop_flag.clone();
    let end_request = Arc::new(AtomicBool::new(false));
    let end_ctrlc = end_request.clone();
    let mut interrupted = false;
    ctrlc::set_handler(move || {
        if interrupted {
            stop_flag_ctrlc.store(true, std::sync::atomic::Ordering::SeqCst);
        } else {
            end_ctrlc.store(true, std::sync::atomic::Ordering::SeqCst);
            interrupted = true;
        }
    })
    .expect("Error setting Ctrl-C handler");

    if let Err(err) = conchordal::app::run_render(
        &args.scenario,
        args.output,
        config,
        stop_flag,
        args.seed,
        args.report.as_deref(),
        args.reserve_runtime_ids_through,
        args.duration,
        end_request,
    ) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
