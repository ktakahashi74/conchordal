let drones = spawn("drones", 2);
for a in drones {
    a.set_amp(0.1);
    a.set_freq(220.0);
}
finish();
