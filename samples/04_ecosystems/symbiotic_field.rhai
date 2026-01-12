print("Symbiotic field");

spawn("kick", 1, #{ body: #{ amp: 0.6 }, pitch: #{ center_hz: 60.0 } });
wait(1.0);
spawn("pad_low", 1, #{ body: #{ amp: 0.3 }, pitch: #{ center_hz: 110.0 } });
spawn("bugs", 5, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 600.0 } });
wait(3.0);
end();
