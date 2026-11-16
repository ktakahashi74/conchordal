use conchordal_web_ui::frame::{Decoder, Encoder};
use std::{io::Read, time::Instant};
/// Replays length-prefixed snapshot packets (little-endian u32 length, then the packet)
/// through one decoder and encoder, asserting byte-identical re-encoding.
fn main() {
    let mut bytes = Vec::new();
    std::fs::File::open(std::env::args().nth(1).expect("frames.bin"))
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let (mut decoder, mut encoder) = (Decoder::default(), Encoder::default());
    let mut rows = Vec::new();
    let mut rest = &bytes[..];
    while !rest.is_empty() {
        let length = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        let packet = &rest[4..4 + length];
        rest = &rest[4 + length..];
        let start = Instant::now();
        let frame = decoder.decode(packet).unwrap();
        let decode = start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        let encoded = encoder.encode(frame).unwrap();
        let encode = start.elapsed().as_secs_f64() * 1000.;
        rows.push(serde_json::json!({"bytes":packet.len(),"decode_ms":decode,"encode_ms":encode}));
        assert_eq!(encoded, packet);
    }
    println!("{}", serde_json::to_string(&rows).unwrap());
}
