let drones_tag = "drones";
spawn(
    drones_tag,
    5,
    #{ method: #{ mode: "harmonicity", min_freq: 200.0, max_freq: 1200.0 } },
    #{ body: #{ amp: 0.15 }, phonation: #{ type: "hold" } }
);

wait(10.0);
release(drones_tag, 3.0);
wait(4.0);
end();
