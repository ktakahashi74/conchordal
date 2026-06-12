print("TEST_OK: note_smoke");
let tone = place(sine(), at(440.0).count(1)).amp(0.5);
flush();
for i in 0..10 {
    wait(1.0);
}
release(tone);
wait(0.2);
