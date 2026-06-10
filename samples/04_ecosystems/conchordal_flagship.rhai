// Conchordal v0.4 flagship etude: "Emergence and Resolution".
//
// A single directed arc, not a feature catalogue. The composer shapes the macro
// drama with two knobs only -- a harmonic_mirror arch (consonant -> dissonant ->
// consonant) and register transposition -- while harmony emerges from the
// agents. Rhythm stays meaningful because every metric voice locks to ONE shared
// emergent attractor (the production meter the population itself drives -- there
// is no imposed clock). meter_stability makes that attractor deep, so the pulse
// forms readily and reads as a load-bearing heartbeat:
//   metric heartbeat -> an accented spine that drives the shared beat hard
//   metric colony    -> a living chord that locks to the SAME emergent beat and
//                       snaps to consonance each onset, so it reads as one with
//                       the heartbeat
//   flow             -> non-metric shimmer that appears only at the tension peak
// The colony's viability + respawn is what lets it survive the dissonant peak
// and reach the resolution: the return of tonal gravity (see the Manifesto).

seed(20260609);

let root_hz = 110.0;

// Hidden harmonic ground: defines the consonance landscape the agents resolve
// toward. Sensed by the ecology, not presented directly.
let field_anchor = harmonic()
    .brain("drone")
    .send(habitat_bus)
    .amp(0.060)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(8))
    .brightness(0.24)
    .sustain_drive(0.0015)
    .adsr(0.7, 1.3, 0.96, 2.6);

// The heartbeat: a strongly-locked accent on the root that drives the shared
// meter hard, so a deep beat attractor forms for everything else to lock onto.
let metric_body = harmonic()
    .amp(0.045)
    .metric()
    .rhythm_role("accent")
    .cycles(2)
    .anchor()
    .modes(odd_modes().count(5))
    .brightness(0.40)
    .adsr(0.006, 0.080, 0.45, 0.18);

// The living colony: fires on the shared pulse grid, climbs toward consonance,
// and survives the dissonant peak through consonance-biased respawn. Its life is
// harmonic (which pitches snap in and survive each onset), not rhythmic drift --
// so it reads as locked to the heartbeat while the chord stays alive.
let consonance_colony = harmonic()
    .amp(0.038)
    .metric()
    .cycles(3)
    .seek_consonance()
    .modes(harmonic_modes().count(7))
    .brightness(0.48)
    .avoid_neighbors(0.60)
    .initial_energy(0.58)
    .energy_cap(0.92)
    .metabolism(0.060)
    .action_cost(0.013)
    .viability_rate(0.24)
    .consonance_viability(0.32, 0.82)
    .dissonance_cost(0.42)
    .respawn_consonance()
    .respawn_capacity(10)
    .respawn_min_c_level(0.26)
    .respawn_background_death_rate(0.009)
    .adsr(0.055, 0.380, 0.70, 0.82);

// Non-metric shimmer: only present at the tension peak, like rain over a
// strained field.
let flow_particles = modal()
    .amp(0.020)
    .flow()
    .cycles(3)
    .seek_consonance()
    .modes(landscape_density_modes().count(6).range(1.2, 5.4).gamma(1.4))
    .brightness(0.72)
    .avoid_neighbors(0.42)
    .adsr(0.024, 0.320, 0.38, 0.74);

section("Conchordal -- Emergence and Resolution", || {
    // I. Genesis: consonant ground rises from silence.
    harmonic_mirror(0.12);
    let root = place(field_anchor, at(root_hz).count(1));
    wait(3.5);

    // II. Pulse: the heartbeat enters alone over the ground.
    let pulse = place(metric_body, at(root_hz).count(1));
    wait(5.5);

    // III. The colony: it fires on the shared pulse grid and climbs toward
    // consonance, while the field slowly bends toward tension.
    let colony = place(consonance_colony, density(80.0, 900.0).count(8).spacing(0.84));
    wait(7.0);
    harmonic_mirror(0.40);
    wait(7.0);

    // IV. Tension peak: dissonant field, lifted register, shimmering flow.
    harmonic_mirror(0.72);
    root.freq(root_hz * 1.5);
    pulse.freq(root_hz * 1.5);
    colony.amp(0.034);
    wait(5.0);
    let flow = place(flow_particles, density(200.0, 1500.0).count(9).spacing(0.66));
    harmonic_mirror(0.88);
    wait(8.0);

    // V. Resolution: the field returns to consonance and the register falls
    // home; agents settle onto stable peaks -- the return of tonal gravity.
    harmonic_mirror(0.45);
    flow.amp(0.014);
    colony.glide(0.22);
    root.freq(root_hz);
    pulse.freq(root_hz);
    wait(5.0);
    release(flow);
    wait(2.0);
    harmonic_mirror(0.10);
    colony.amp(0.028);
    wait(8.0);
    release(colony);
    wait(3.0);
    release(pulse);
    wait(2.0);
    release(root);
    wait(6.0);
});
