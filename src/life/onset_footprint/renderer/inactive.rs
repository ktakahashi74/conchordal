use super::{Recipe, Tone};

#[inline(always)]
pub(super) fn legacy(recipe: &Recipe) -> bool {
    let _ = recipe.renderer_phase3;
    true
}

#[inline(always)]
pub(super) fn supported(_: &Recipe) -> bool {
    true
}

#[inline(always)]
pub(super) fn enable(_: &Recipe, _: &mut Tone) {}
