let tone = place(sine(), at(440.0).count(1)).amp(0.1).sustain();
wait(2.0);
release(tone);
wait(0.5);
