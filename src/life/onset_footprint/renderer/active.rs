use super::{Recipe, Tone};

pub(super) fn legacy(recipe: &Recipe) -> bool {
    !recipe.renderer_phase3
}

pub(super) fn supported(recipe: &Recipe) -> bool {
    recipe.body.supports_body_models(recipe.renderer_phase3)
}

pub(super) fn enable(recipe: &Recipe, tone: &mut Tone) {
    if recipe.renderer_phase3 {
        tone.enable_phase3();
    }
}
