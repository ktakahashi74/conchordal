let bed = derive(sine)
    .amp(0.05)
    .phonation("hold")
    .metabolism(0.0)
    .sustain_drive(0.01)
    .pitch_mode("lock");

let mover = derive(sine)
    .amp(0.05)
    .phonation("hold")
    .metabolism(0.0)
    .sustain_drive(0.012)
    .pitch_mode("free")
    .pitch_core("hill_climb")
    .proposal_interval(0.60)
    .global_peaks(6, 60.0)
    .ratio_candidates(4)
    .move_cost(0.65)
    .improvement_threshold(0.12)
    .move_cost_time_scale("proposal_interval")
    .leave_self_out_harmonics(1)
    .exploration(0.10)
    .persistence(0.72)
    .crowding(1.20)
    .pitch_apply("glide")
    .pitch_glide(0.40);

scene("HillClimb Stable Convergence", || {
    create(bed, 1).freq(110.0);
    create(bed, 1).freq(165.0);
    create(bed, 1).freq(220.0);
    wait(0.6);

    // place(...) sets lock mode, so switch back to free for runtime search.
    let movers = create(mover, 6)
        .place(consonance_density_pmf(90.0, 420.0))
        .mode("free");

    wait(28.0);
    release(movers);
    wait(0.6);
});
