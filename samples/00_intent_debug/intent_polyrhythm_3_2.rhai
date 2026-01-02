for i in 0..6 {
intent(440.0, 0.0, 0.08, 0.4, #{ tag: "three" });
intent(660.0, 0.0, 0.08, 0.4, #{ tag: "two" });
intent(440.0, 0.333, 0.08, 0.4, #{ tag: "three" });
intent(660.0, 0.5, 0.08, 0.4, #{ tag: "two" });
intent(440.0, 0.666, 0.08, 0.4, #{ tag: "three" });
    wait(1.0);
}
end();
