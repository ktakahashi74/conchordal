use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
pub struct Args {
    /// Play audio in realtime
    #[arg(long, default_value_t = true, num_args = 0..=1, default_missing_value = "true")]
    pub play: bool,

    /// Scenario path (.rhai only)
    #[arg(value_name = "SCENARIO_PATH")]
    pub scenario_path: String,

    /// Path to config TOML
    #[arg(long)]
    pub config: Option<String>,

    /// Wait for user action to exit after playback (overrides config)
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub wait_user_exit: Option<bool>,

    /// Wait for user action before starting playback (overrides config)
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    pub wait_user_start: Option<bool>,

    /// Run without GUI (headless)
    #[arg(long, default_value_t = false)]
    pub nogui: bool,

    /// Compile scenario script only (no GUI, no audio, no execution)
    #[arg(long, default_value_t = false)]
    pub compile_only: bool,

    /// Write machine-readable runtime report as JSONL
    #[arg(long)]
    pub report: Option<String>,

    /// Save worker hop timings and device counters as JSON after execution (no audio)
    #[arg(long, requires = "nogui", conflicts_with = "compile_only")]
    pub profile: Option<String>,

    /// Request a graceful end at this score time, followed by the release fades
    #[arg(long, value_name = "SEC", value_parser = parse_duration)]
    pub duration: Option<f64>,

    #[arg(skip)]
    pub end_request: std::sync::Arc<std::sync::atomic::AtomicBool>,

    /// Print one bounded runtime telemetry summary per second (no audio recording)
    #[arg(long)]
    pub telemetry: bool,

    /// Override the fresh scenario seed (a script-level seed() still wins)
    #[arg(long)]
    pub seed: Option<u64>,
}

pub fn parse_duration(value: &str) -> Result<f64, String> {
    let duration: f64 = value
        .parse()
        .map_err(|_| "duration must be finite and non-negative")?;
    if !duration.is_finite() || duration < 0.0 {
        return Err("duration must be finite and non-negative".into());
    }
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn help_includes_config_flag() {
        let mut cmd = Args::command();
        let mut help = Vec::new();
        cmd.write_long_help(&mut help).expect("write help");
        let help = String::from_utf8(help).expect("utf8 help");
        assert!(help.contains("--config"));
        assert!(help.contains("--telemetry"));
    }

    #[test]
    fn telemetry_is_opt_in() {
        assert!(
            !Args::try_parse_from(["conchordal", "sample.rhai"])
                .unwrap()
                .telemetry
        );
        assert!(
            Args::try_parse_from(["conchordal", "sample.rhai", "--telemetry"])
                .unwrap()
                .telemetry
        );
    }
    #[test]
    fn wait_user_exit_flag_enables_wait_and_accepts_explicit_false() {
        let args = Args::try_parse_from(["conchordal", "sample.rhai", "--wait-user-exit"]).unwrap();
        assert_eq!(args.wait_user_exit, Some(true));
        let args = Args::try_parse_from(["conchordal", "--wait-user-exit", "sample.rhai"]).unwrap();
        assert_eq!(args.wait_user_exit, Some(true));
        let args =
            Args::try_parse_from(["conchordal", "sample.rhai", "--wait-user-exit=false"]).unwrap();
        assert_eq!(args.wait_user_exit, Some(false));
    }
}
