let tone = create(sine, 1).freq(440.0).amp(0.1).sustain();
wait(2.0);
release(tone);
wait(0.5);
