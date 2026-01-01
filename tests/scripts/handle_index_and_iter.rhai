print("TEST_OK: handle_index_and_iter");
let drones = spawn("drones", 5);
drones[0].set_freq(220.0);
for a in drones {
    a.set_amp(0.05);
}
finish();
