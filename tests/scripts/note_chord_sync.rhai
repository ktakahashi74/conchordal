print("TEST_OK: note_chord_sync");
let chord = place(sine("sine"), at(440.0).count(3)).amp(0.4);
flush();
wait(1.2);
release(chord);
wait(0.2);
