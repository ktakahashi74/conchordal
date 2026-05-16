print("TEST_OK: stable_order_for_loop_expansion");
for i in 0..4 {
    place(sine("sine"), at(440.0).count(1)).amp(0.1 + i * 0.05);
}
flush();
wait(0.2);
