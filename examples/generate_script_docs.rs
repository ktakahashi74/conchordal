use std::fs;
use std::path::Path;

use rhai::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::new();

    let docs = rhai_autodocs::export::options()
        .include_standard_packages(false)
        .export(&engine)?;

    let markdown = rhai_autodocs::generate::mdbook().generate(&docs)?;

    let out_dir = Path::new("docs/rhai_book/src/reference");
    fs::create_dir_all(out_dir)?;

    for (name, content) in markdown {
        let content = content.replace("</br>", "<br/>");
        let path = out_dir.join(format!("{name}.md"));
        fs::write(path, content)?;
    }

    Ok(())
}
