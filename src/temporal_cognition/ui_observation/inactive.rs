use conchordal_web_ui::frame::Diagnostic;

#[derive(Clone, Debug, Default)]
pub(crate) struct TemporalUi;

impl TemporalUi {
    #[inline(always)]
    pub(crate) fn diagnostics(&self) -> Vec<Diagnostic> {
        Vec::new()
    }
}
