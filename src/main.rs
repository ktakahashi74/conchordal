use clap::Parser;
use conchordal::{app, cli::Args, config::AppConfig};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tracing_subscriber::EnvFilter;

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
    let mut config = match AppConfig::load_or_default(args.config.as_deref()) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("{err:#}");
            std::process::exit(1);
        }
    };
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

    if args.compile_only {
        if let Err(err) = app::run_compile_only(args) {
            eprintln!("{err}");
            std::process::exit(1);
        }
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
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 900.0]),
        ..Default::default()
    };

    let runtime_failed = Arc::new(AtomicBool::new(false));
    let gui_failed = Arc::clone(&runtime_failed);
    let result = eframe::run_native(
        "Conchordal",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(app::App::new(
                cc,
                args.clone(),
                config.clone(),
                stop_flag.clone(),
                Arc::clone(&gui_failed),
            )))
        }),
    )
    .map_err(|e| {
        eprintln!("Error: {:?}", e);
        e
    });
    if runtime_failed.load(Ordering::SeqCst) {
        std::process::exit(1);
    }
    result
}
