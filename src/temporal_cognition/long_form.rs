//! Offline material bookkeeping; these labels never enter auditory inference.

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) struct Material {
    pub initial_motif: [f64; 2],
    pub returns: [[f64; 2]; 3],
    pub target_end_max: f64,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            initial_motif: [0., 8.],
            returns: [[144., 152.], [804., 812.], [1740., 1748.]],
            target_end_max: 11.,
        }
    }
}

impl Material {
    pub fn load() -> Self {
        let Some(path) = std::env::var_os("CONCHORDAL_I9_MATERIAL") else {
            return Self::default();
        };
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let c = &v["construction"];
        let material = Self {
            initial_motif: serde_json::from_value(c["initial_motif"].clone()).unwrap(),
            returns: serde_json::from_value(c["returns"].clone()).unwrap(),
            target_end_max: v["reference_target_end_max_seconds"].as_f64().unwrap(),
        };
        assert!(material.initial_motif[0] >= 0.);
        let mut prior_end = material.initial_motif[1];
        assert!(prior_end > material.initial_motif[0]);
        assert!(material.target_end_max.is_finite() && material.target_end_max >= prior_end);
        for [start, end] in material.returns {
            assert!(start.is_finite() && end.is_finite() && start >= prior_end && end > start + 2.);
            prior_end = end;
        }
        material
    }

    pub fn is_target(&self, start: f64, end: f64) -> bool {
        start >= self.initial_motif[0]
            && start < self.initial_motif[1]
            && end <= self.target_end_max
    }
}
