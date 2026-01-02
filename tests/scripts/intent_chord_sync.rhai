print("TEST_OK: intent_chord_sync");
for i in 0..8 {
    intent(440.0, 1.0, 0.15, 0.4, #{ tag: "chord" });
    intent(550.0, 1.0, 0.15, 0.4, #{ tag: "chord" });
    intent(660.0, 1.0, 0.15, 0.4, #{ tag: "chord" });
    wait(1.0);
}
end();
