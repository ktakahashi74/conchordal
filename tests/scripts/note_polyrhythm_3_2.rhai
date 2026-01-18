print("TEST_OK: note_polyrhythm_3_2");
let three = create(sine, 1).freq(440.0).amp(0.4);
let two = create(sine, 1).freq(660.0).amp(0.4);
flush();
wait(1.0);
release(three);
release(two);
wait(0.2);
