print("TEST_OK: tag_selector_ops");
let drones = create(saw, 2).amp(0.1);
flush();
wait(0.5);
release(drones);
wait(0.5);
