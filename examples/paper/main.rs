mod e4_abcd_analyze;
mod paper_plots;
mod sim;

use std::env;

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if let Some(pos) = args.iter().position(|a| a == "--e4-abcd-analyze") {
        args.remove(pos);
        if let Err(err) = e4_abcd_analyze::run_from_args(&args) {
            eprintln!("e4 abcd analyze failed: {err}");
            std::process::exit(1);
        }
        return;
    }

    if let Err(err) = paper_plots::main() {
        eprintln!("paper plots failed: {err}");
        std::process::exit(1);
    }
}
