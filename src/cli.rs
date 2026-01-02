use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
pub struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true")]
    pub play: bool,

    /// Write audio to wav file
    #[arg(long)]
    pub wav: Option<String>,

    /// Scenario path (.rhai only)
    #[arg(value_name = "SCENARIO_PATH")]
    pub scenario_path: String,

    /// Path to config TOML
    #[arg(long, default_value = "config.toml")]
    pub config: String,

    /// Wait for user action to exit after playback (overrides config)
    #[arg(long, num_args = 0..=1, default_missing_value = "false")]
    pub wait_user_exit: Option<bool>,

    /// Wait for user action before starting playback (overrides config)
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub wait_user_start: Option<bool>,

    /// Run without GUI (headless)
    #[arg(long, default_value_t = false)]
    pub nogui: bool,

    /// Render intent board audio only
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub intent_only: Option<bool>,

    /// Compile scenario script only (no GUI, no audio, no execution)
    #[arg(long, default_value_t = false)]
    pub compile_only: bool,
}
