// E3 demo (conceptual sketch): selection pressure using consonance density.
let anchor = derive(sine).amp(0.07).phonation("hold");
let agent = derive(sine).amp(0.03).phonation("hold");

scene("E3 Metabolic Selection Demo", || {
    let base = create(anchor, 1).freq(82.41);
    flush();
    wait(0.6);

    for i in 0..6 {
        let strat = consonance_density(82.41, 220.0);
        create(agent, 1).place(strat);
        wait(0.2);
    }

    wait(1.5);
    release(base);
    wait(0.3);
});
