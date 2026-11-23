use super::*;

pub(crate) fn supports_body_model(
    control: &VoiceControl,
    snapshot: &BodySnapshot,
    renderer_phase3: bool,
) -> bool {
    model_kind(control, renderer_phase3) == Some(snapshot.kind)
}

pub(crate) fn model_kind(control: &VoiceControl, renderer_phase3: bool) -> Option<BodyKind> {
    let registry = global_factory_registry()
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let method = body_method_id(control.body.method);
    registry.factories.get(method).map_or_else(
        || HarmonicBodyFactory.model_kind(renderer_phase3),
        |factory| factory.model_kind(renderer_phase3),
    )
}
