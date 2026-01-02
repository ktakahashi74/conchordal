use conchordal::core::timebase::Timebase;
use conchordal::life::intent::{Intent, IntentBoard};
use conchordal::life::intent_renderer::IntentRenderer;

#[test]
fn smoke_intent_board_and_renderer() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut board = IntentBoard::new(tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    board.publish(Intent {
        source_id: 0,
        intent_id: 1,
        onset: 10,
        duration: 5,
        freq_hz: 440.0,
        amp: 0.2,
        tag: None,
        confidence: 1.0,
        body: None,
    });
    let hits: Vec<_> = board.query_range(0..32).collect();
    assert_eq!(hits.len(), 1);

    let mut renderer = IntentRenderer::new(tb);
    let out = renderer.render(&board, 0);
    assert_eq!(out.len(), tb.hop);
}
