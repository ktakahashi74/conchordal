// DEMO-B2 draft by the orchestration lead, pending author adoption.
// Built only from parts the author heard as music: Flora at pace 1, rain with 1/f sway,
// and held, anchored fields. No Voice searches for pitch; no progression is written.
// The first field's pitch is a founder choice. The second field is placed by the ecology
// (consonance placement, then held). Sections, ranges, amplitudes, the rain's mean rate and
// sway depth, Flora pace and the roughness-aversion arc are the drafter's composer choices
// for the author to reshape.

output_gain(4.5);
temporal_basin(1.6, 2.4);

let root_hz = 110.0;

let field = harmonic().brain("drone").sustain().anchor()
    .modes(harmonic_modes().count(8)).amp(0.045).brightness(0.32)
    .adsr(6.0, 1.0, 0.95, 8.0);
let heard_field = harmonic().brain("drone").sustain().anchor()
    .modes(harmonic_modes().count(8)).amp(0.03).brightness(0.28)
    .adsr(14.0, 1.0, 0.95, 10.0);
let second_field = harmonic().brain("drone").sustain().anchor()
    .modes(harmonic_modes().count(8)).amp(0.028).brightness(0.36)
    .adsr(10.0, 1.0, 0.95, 10.0);
let moss = flora().amp(0.028).fade(10.0).pace(1.0).pitch_apply_mode("rearticulate");
// The rain's strength follows its 1/f sway alone; only its entry and exit are written
// (author, 2026-11-14). Two octaves of sway let it thin almost to nothing and return.
// rate 6 keeps the previous draft's mean density (mean of 2^(sigma*g) is 2.61 at sigma 2).
// Drops that land on more consonant places ring longer (ring), read from the field without
// the rain's own drops (leave_drops_out); objects take the drops' places once silent
// (objects_follow_drops). width(0.0052) keeps each drop's heard pitch inside the landscape bin
// it was drawn from, and integer-ratio partials let that alignment be heard (RW-8..RW-12).
let rainfall = rain().amp(0.012).rate(6).sway(2).objects(4).fade(8)
    .leave_drops_out(true).objects_follow_drops(true).ring(2)
    .width(0.0052).impact_modes(harmonic_modes()).object_modes(harmonic_modes());
let lichen = flora().tone("resonant").amp(0.018).fade(12.0).pace(1.0)
    .pitch_apply_mode("rearticulate");

// The field is first sensed only by the ecology; the moss grows on it unheard.
// The short wait lets the field exist before the moss reads the terrain.
let hidden = place(field.send(habitat_bus), at(root_hz));
wait(2.0);

let ground = place(moss, consonance(root_hz, 880.0));
let shower = place(rainfall, consonance(150, 3520));
section("germination", || { wait(50.0); });

// What the ecology has been hearing becomes audible.
let revealed = place(heard_field.send(presentation_bus), at(root_hz));
section("reveal", || { ground.amp(0.034); wait(35.0); });

let canopy = place(lichen, consonance(440.0, 3520.0));
section("canopy", || {
    set_roughness_aversion(0.6);
    wait(50.0);
});

// A second held ground, at a place the ecology finds consonant.
let other = place(second_field, consonance(196.0, 660.0));
section("second ground", || { canopy.amp(0.026); wait(30.0); });

section("thicket", || {
    canopy.amp(0.034);
    ground.pace(2.0);
    canopy.pace(2.0);
    set_roughness_aversion(0.25);
    wait(45.0);
});

section("break", || {
    canopy.amp(0.008);
    ground.amp(0.014);
    wait(10.0);
});

section("clearing", || {
    set_roughness_aversion(1.6);
    ground.pace(1.0);
    canopy.pace(1.0);
    ground.amp(0.036);
    canopy.amp(0.02);
    wait(45.0);
});

release(canopy);
release(other);
section("after", || {
    set_roughness_aversion(1.0);
    ground.amp(0.024);
    wait(35.0);
});

release(shower);
release(revealed);
section("last light", || { ground.amp(0.016); wait(20.0); });

release(ground);
release(hidden);
wait(12.0);
