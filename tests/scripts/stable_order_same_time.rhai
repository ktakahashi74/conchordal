print("TEST_OK: stable_order_same_time");
for i in 0..3 {
    place(sine(), at(200.0 + i * 50.0).count(1));
}
flush();
wait(0.2);
