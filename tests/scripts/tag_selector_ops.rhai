print("TEST_OK: tag_selector_ops");
let drones = place(saw("saw"), at(440.0).count(2)).amp(0.1);
flush();
wait(0.5);
release(drones);
wait(0.5);
