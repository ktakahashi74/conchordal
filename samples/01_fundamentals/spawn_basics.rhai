// Mixed strategies in one sweep (quick listening tour)
spawn("rnd", 3, #{ body: #{ amp: 0.18 }, pitch: #{ center_hz: 400.0 } }); wait(0.9);
spawn("harm", 2, #{ body: #{ amp: 0.22 }, pitch: #{ center_hz: 520.0 } }); wait(0.8);
spawn("gap", 2, #{ body: #{ amp: 0.18 }, pitch: #{ center_hz: 360.0 } }); wait(0.9);
spawn("hdens", 3, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 300.0 } }); wait(0.8);
spawn("zero", 2, #{ body: #{ amp: 0.18 }, pitch: #{ center_hz: 420.0 } }); wait(0.8);
spawn("lowharm", 2, #{ body: #{ amp: 0.18 }, pitch: #{ center_hz: 280.0 } });
wait(1.5);
end();
