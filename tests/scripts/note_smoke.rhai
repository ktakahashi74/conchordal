print("TEST_OK: note_smoke");
let tone = create(sine, 1).freq(440.0).amp(0.5);
flush();
for i in 0..10 {
    wait(1.0);
}
release(tone);
wait(0.2);
