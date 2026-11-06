use conchordal_web_ui::frame::UiFrame;
use std::{
    io::{BufRead, BufReader},
    time::Instant,
};
fn main() {
    let file = std::fs::File::open(std::env::args().nth(1).expect("frames.jsonl")).unwrap();
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        let bytes = line.unwrap();
        let start = Instant::now();
        let frame = UiFrame::decode(bytes.as_bytes()).unwrap();
        let decode = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let encoded = frame.encode().unwrap();
        let encode = start.elapsed().as_secs_f64() * 1000.;
        rows.push(serde_json::json!({"bytes":bytes.len(),"decode_ms":decode,"encode_ms":encode}));
        assert_eq!(encoded, bytes.as_bytes());
    }
    println!("{}", serde_json::to_string(&rows).unwrap());
}
