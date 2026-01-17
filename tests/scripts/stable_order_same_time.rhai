print("TEST_OK: stable_order_same_time");
for i in 0..3 {
    create(sine, 1).freq(200.0 + i * 50.0);
}
flush();
wait(0.2);
