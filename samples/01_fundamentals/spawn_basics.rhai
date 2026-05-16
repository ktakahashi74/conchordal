let base = harmonic("harmonic").amp(0.1).sustain();

place(base, peaks(220.0).range(1.0, 4.0).spacing(1.0).count(4));
wait(2.0);

place(base, density(110.0, 880.0).count(4));
wait(2.0);

place(base, random(110.0, 880.0).count(4));
wait(2.0);

place(base, line(110.0, 880.0).count(4));
wait(2.0);
