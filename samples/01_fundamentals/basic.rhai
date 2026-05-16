let drones = place(sine("sine"), at(440.0).count(3)).amp(0.08).sustain();
drones.freq(220.0);
wait(2.0);
release(drones);
wait(0.5);
