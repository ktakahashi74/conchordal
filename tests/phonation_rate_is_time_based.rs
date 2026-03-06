use conchordal::life::phonation_engine::{
    AccumulatorInterval, CoreState, IntervalInput, PhonationInterval,
};

#[test]
fn phonation_rate_is_time_based() {
    let mut interval = AccumulatorInterval::new(1.0, 0, 1);
    let state = CoreState { is_alive: true };
    let dt_sec = 1.0 / 12.0;
    let tick_step = 10u64;
    let mut onsets = 0;
    for gate in 0..24u64 {
        let input = IntervalInput {
            gate,
            tick: gate * tick_step,
            dt_theta: 1.0,
            dt_sec,
            weight: 1.0,
        };
        if interval.on_candidate(&input, &state).is_some() {
            onsets += 1;
        }
    }
    assert!(
        (1..=3).contains(&onsets),
        "expected 1..=3 onsets, got {onsets}"
    );
}
