seed(1);
let drones = create(sine, 3).amp(0.15).phonation("hold");
drones.freq(220.0);
flush();
wait(2.0);
release(drones);
wait(0.5);
