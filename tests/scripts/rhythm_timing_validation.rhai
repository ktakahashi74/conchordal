print("TEST_OK: rhythm_timing_validation");
let metro = place(sine(), at(440.0).count(1)).amp(0.6).repeat();
flush();
for i in 0..4 {
    wait(0.25);
}
release(metro);
wait(0.2);
