print("TEST_OK: note_microtiming");
let micro = place(sine("sine"), at(440.0).count(3)).amp(0.4);
flush();
for i in 0..6 {
    wait(1.2);
}
release(micro);
wait(0.2);
