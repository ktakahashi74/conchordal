// A single moving Voice changes the same sound that the listener observes.
// Pitch references are sustained; no separate beat carrier or forced resolution.
seed(20260905);
let ground = harmonic().brain("drone").amp(0.035).sustain().anchor()
    .modes(harmonic_modes().count(8)).sustain_drive(0.002);
place(ground, at(110.0));
place(ground, at(165.0));
wait(0.5);
let seeker = harmonic().amp(0.04).sustain().seek_consonance()
    .glide(0.1).temperature(0.0).proposal_interval(0.05)
    .endurance(120.0);
place(seeker, random(200.0, 500.0));
wait(20.0);
