print("TEST_OK: consonance_density_reversed_range");
let base = harmonic().amp(0.08).sustain();
place(base, density(880.0, 110.0).count(4));
flush();
wait(0.2);
