use super::ArrivalContext;
use crate::life::voice::Voice;

#[derive(Default)]
pub(crate) struct ArrivalSources<'a>(std::marker::PhantomData<&'a ()>);

impl<'a> ArrivalSources<'a> {
    #[inline(always)]
    pub(crate) fn for_voice(&self, _: &Voice, _: u64) -> Option<ArrivalContext<'a>> {
        None
    }
}
