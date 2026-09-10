seed(21);

let actor = place(sine().amp(0.12).sustain().anchor().entrained()
    .cycles(1).send(presentation_bus), at(330.0));
wait(2.0);
let environment = place(sine().amp(0.10).sustain().anchor()
    .send(habitat_bus), at(737.0));
wait(2.0);
release(environment);
wait(1.0);
release(actor);
wait(0.25);
print("TEST_OK: participation_outcome");
