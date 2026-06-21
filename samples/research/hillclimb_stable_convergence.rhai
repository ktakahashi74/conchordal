// Movement-mechanism assay.
// Movers converge against a stable harmonic bed with explicit hill-climb tuning.

seed(20260402);

let bed = sine()
    .amp(0.05)
    .sustain()
    .metabolism(0.0)
    .sustain_drive(0.01)
    .anchor();

let mover = sine()
    .amp(0.05)
    .sustain()
    .metabolism(0.0)
    .sustain_drive(0.012)
    .seek_consonance()
    .proposal_interval(0.60)
    .global_peaks(6, 60.0)
    .ratio_candidates(4)
    .move_cost(0.65)
    .move_cost_time_scale("proposal_interval")
    .leave_self_out_harmonics(1)
    .temperature(0.1)
    .avoid_neighbors(1.20)
    .glide(0.40);

section("HillClimb Stable Convergence", || {
    place(bed, at(110.0).count(1));
    place(bed, at(165.0).count(1));
    place(bed, at(220.0).count(1));
    wait(0.6);

    let movers = place(mover, consonance(90.0, 420.0).count(6));

    wait(28.0);
    release(movers);
    wait(0.6);
});
