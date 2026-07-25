use conchordal::life::phonation_engine::{CoreState, IntervalInput, OnsetRule};

#[test]
fn phonation_rate_is_time_based() {
    let mut interval = OnsetRule::accumulator(1.0, 0, 1);
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let dt_sec = 1.0 / 12.0;
    let tick_step = 10u64;
    let mut onsets = 0;
    for gate in 0..24u64 {
        let input = IntervalInput {
            gate,
            tick: gate * tick_step,
            dt_sec,
            weight: 1.0,
        };
        if interval.on_candidate(&input, &state).is_some() {
            onsets += 1;
        }
    }
    // rate 1.0 Hz over 24 gates of dt=1/12 s is deterministic: exactly 2 onsets.
    // A range here would also pass if the rate drifted by 50%.
    assert_eq!(onsets, 2, "expected exactly 2 onsets, got {onsets}");
}
