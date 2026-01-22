let base = derive(harmonic).amp(0.1).phonation("hold");

create(base, 4)
    .place(consonance(220.0).range(1.0, 4.0).min_dist(1.0));
wait(2.0);

create(base, 4).place(consonance_density(110.0, 880.0));
wait(2.0);

create(base, 4).place(random_log(110.0, 880.0));
wait(2.0);

create(base, 4).place(linear(110.0, 880.0));
wait(2.0);
