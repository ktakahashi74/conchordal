print(">>> Experiment: Self-Organizing Rhythm <<<");

spawn("starter", 1, #{ body: #{ amp: 0.4 }, pitch: #{ center_hz: 100.0 } });
wait(2.0);

scene("Phase 2: Recruitment");
for i in 0..4 {
    spawn("follower", 4, #{ body: #{ amp: 0.08 }, pitch: #{ center_hz: 400.0 } });
    wait(1.0);
}

wait(4.0);
end();
