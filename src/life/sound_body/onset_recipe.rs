use crate::life::sound::{BodyKind, BodySnapshot};
use sha2::{Digest, Sha256};

impl BodySnapshot {
    pub(crate) fn hash_into(&self, hasher: &mut Sha256) {
        // Preserve the byte encoding of representative-onset recipe identities.
        hasher.update([match self.kind {
            BodyKind::Sine => 0u8,
            BodyKind::Harmonic => 1,
            BodyKind::Modal => 2,
        }]);
        for value in [
            self.amp_scale,
            self.brightness,
            self.inharmonic,
            self.spread,
            self.motion,
        ] {
            hasher.update(value.to_bits().to_le_bytes());
        }
        if self.attack_noise > 0. {
            hasher.update(b"attack_noise");
            hasher.update(self.attack_noise.to_le_bytes());
        }
        hasher.update((self.unison as u64).to_le_bytes());
        match self.ratios.as_deref() {
            None => hasher.update([0u8]),
            Some(ratios) => {
                hasher.update([1u8]);
                hasher.update((ratios.len() as u64).to_le_bytes());
                for ratio in ratios {
                    hasher.update(ratio.to_bits().to_le_bytes());
                }
            }
        }
    }

    pub(crate) fn has_same_structure(&self, other: &Self) -> bool {
        self.kind == other.kind && self.unison == other.unison && self.ratios == other.ratios
    }
}
