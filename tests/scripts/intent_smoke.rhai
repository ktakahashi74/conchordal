print("TEST_OK: intent_smoke");
for i in 0..10 {
    intent(440.0, 0.8, 0.10, 0.5, #{ tag: "smoke" });
    wait(1.0);
}
end();
