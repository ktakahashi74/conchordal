let drones = create(sine, 3).energy(0.08).sustain();
drones.freq(220.0);
wait(2.0);
release(drones);
wait(0.5);
