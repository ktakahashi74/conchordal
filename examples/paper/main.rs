mod sim;
mod paper_plots;

fn main() {
    if let Err(err) = paper_plots::main() {
        eprintln!("paper plots failed: {err}");
        std::process::exit(1);
    }
}
