// Deterministic endurance contract assay.
// Run with --report and compare configured_endurance_sec,
// energy_depletion_sec, and lifetime_sec on each death record.

seed(20260710);

let baseline = sine()
    .amp(0.025)
    .sustain()
    .anchor()
    .endurance(2.0)
    .attack_cost_fraction(0.0)
    .attack_recharge_fraction(0.0)
    .dissonance_penalty(0.0)
    .adsr(0.01, 0.05, 0.80, 0.20);

section("Lifecycle Time-Domain Assay", || {
    place(baseline, at(220.0));
    place(variant(baseline).endurance(4.0), at(330.0));
    place(variant(baseline).endurance(8.0), at(440.0));
    wait(9.0);
});
