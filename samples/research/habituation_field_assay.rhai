// Habituation field assay — deterministic, seeded.
// Verifies production-loop closure: mobile consonance-seeking voices erode the
// basin they occupy, depart, the basin recovers after they leave, and voices
// re-approach. A persistent anchored terrain keeps the raw attractors alive so
// closure is even possible (a lone voice that is its own only source cannot
// close the loop).
//
// Enable habituation via config (see tests/habituation_field_assay.rs):
//   [psychoacoustics.habituation]
//   enabled = true
// With habituation disabled this scenario settles (causal control).
seed(20260717);

// Persistent shared terrain: anchored sustained harmonics on the habitat bus
// create consonant basins that survive any single mobile voice leaving.
let terrain = harmonic()
    .send(habitat_bus)
    .amp(0.04)
    .sustain()
    .anchor()
    .modes(harmonic_modes().count(6))
    .brightness(0.30);

// Mobile consonance-seeking population: erodes the basin it occupies, then
// (if the loop closes) moves off, lets it recover, and re-approaches.
let mover = harmonic()
    .amp(0.035)
    .entrained()
    .cycles(3)
    .seek_consonance()
    .endurance(12.0)
    .recovery(4.0)
    .attack_cost_fraction(0.02)
    .attack_recharge_fraction(0.50)
    .consonance_viability(0.25, 0.85)
    .dissonance_penalty(0.40)
    .adsr(0.04, 0.20, 0.72, 0.45);

section("Habituation Field Assay", || {
    place(terrain, at(110.0));
    place(variant(terrain).brightness(0.24), at(165.0));
    place(mover, consonance(120.0, 880.0).count(5));
    wait(60.0);
});
