print("TEST_OK: note_polyrhythm_3_2");
let three = place(sine(), at(440.0).count(1)).amp(0.4);
let two = place(sine(), at(660.0).count(1)).amp(0.4);
flush();
wait(1.0);
release(three);
release(two);
wait(0.2);
