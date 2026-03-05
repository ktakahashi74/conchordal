print("TEST_OK: consonance_density_reversed_range");
let base = derive(harmonic).energy(0.08).sustain();
create(base, 4).place(consonance_density_pmf(880.0, 110.0));
flush();
wait(0.2);
