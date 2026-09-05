seed(21);

// Keep listener evidence independent of the moving habitat-only Voice.
let reference = harmonic().amp(0.08).send(presentation_bus).sustain().anchor();
place(reference, at(220.0));
place(reference, at(235.0));
place(reference, at(330.0));
let ground = harmonic().amp(0.05).send(habitat_bus).sustain().anchor();
place(ground, at(110.0));
wait(0.5);

let seeker = harmonic().amp(0.04).send(habitat_bus).sustain()
    .seek_consonance().glide(0.1).temperature(0.0).proposal_interval(0.05);
place(seeker, random(200.0, 500.0));
wait(8.0);

print("TEST_OK: dcc_pitch_feedback");
