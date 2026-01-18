print("TEST_OK: note_microtiming");
let micro = create(sine, 3).freq(440.0).amp(0.4);
flush();
for i in 0..6 {
    wait(1.2);
}
release(micro);
wait(0.2);
