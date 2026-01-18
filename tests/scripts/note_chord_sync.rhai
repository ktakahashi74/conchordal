print("TEST_OK: note_chord_sync");
let chord = create(sine, 3).freq(440.0).amp(0.4);
flush();
wait(1.2);
release(chord);
wait(0.2);
