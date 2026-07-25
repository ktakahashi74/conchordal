let lead = place(sine(), at(440.0).count(1));
flush();
wait(0.2);
release(lead);
wait(0.2);

print("TEST_OK: empty_life_map_ok");
