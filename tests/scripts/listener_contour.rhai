seed(21);

let hidden = place(sine().amp(0.06).sustain().anchor().send(habitat_bus), at(737.0));
wait(0.6);
let voice = sine().amp(0.12).sustain().anchor().send(presentation_bus)
    .adsr(0.012, 0.01, 1.0, 0.025);
let pitches = [220.0, 252.7, 311.1, 274.6, 359.8, 335.9, 239.1, 294.3];
for cycle in 0..7 {
    for frequency in pitches {
        let note = place(voice, at(frequency));
        wait(0.195);
        release(note);
        wait(0.165);
    }
}
release(hidden);
wait(1.2);
print("TEST_OK: listener_contour");
