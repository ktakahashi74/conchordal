use conchordal::core::landscape::Landscape;
use conchordal::core::log2space::Log2Space;
use conchordal::core::timebase::Timebase;
use conchordal::life::individual::AgentMetadata;
use conchordal::life::intent_renderer::IntentRenderer;
use conchordal::life::population::Population;
use conchordal::life::scenario::{IndividualConfig, LifeConfig};
use conchordal::life::world_model::WorldModel;

#[test]
fn agents_publish_intents_and_render_audio() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut world = WorldModel::new(tb);
    let mut pop = Population::new(tb.fs);
    let agent_cfg = IndividualConfig {
        freq: 440.0,
        amp: 0.4,
        life: LifeConfig::default(),
        tag: None,
    };
    let metadata = AgentMetadata {
        id: 1,
        tag: None,
        group_idx: 0,
        member_idx: 0,
    };
    let agent = agent_cfg.spawn(1, 0, metadata, tb.fs, 0);
    pop.add_individual(agent);

    let space = Log2Space::new(20.0, 20_000.0, 24);
    let landscape = Landscape::new(space);

    world.advance_to(0);
    pop.publish_intents(&mut world, &landscape, 0);
    assert!(world.board.len() > 0);

    let mut renderer = IntentRenderer::new(tb);
    let out = renderer.render(&world.board, 0);
    assert!(out.iter().any(|s| s.abs() > 1e-6));
}
