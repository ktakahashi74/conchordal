//! PPM-Decay, ppm v0.1.1: escape A, interpolation, no exclusions.
//! Numerical reference: scripts/ppm_decay_reference.py (Harrison et al., 2020).
use std::collections::HashMap;

pub(crate) const MAX_INTERVALS: usize = 4800;
const EMPTY: usize = usize::MAX;

#[derive(Clone, Debug)]
pub(crate) struct Parameters {
    pub(crate) buffer_time: f64,
    pub(crate) buffer_items: usize,
    pub(crate) buffer_weight: f64,
    pub(crate) stm_weight: f64,
    pub(crate) stm_duration: f64,
    pub(crate) ltm_weight: f64,
    pub(crate) half_life: f64,
    pub(crate) asymptote: f64,
    pub(crate) noise: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            buffer_time: 1000.0,
            buffer_items: 15,
            buffer_weight: 1.0,
            stm_weight: 1.0,
            stm_duration: 0.0,
            ltm_weight: 0.6,
            half_life: 3.5,
            asymptote: 0.0,
            noise: 0.8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Trace {
    end: usize,
    next: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct PpmDecay {
    pub(crate) symbols: Vec<u16>,
    pub(crate) times: Vec<f64>,
    traces: Vec<Trace>,
    grams: HashMap<[u16; 5], (usize, usize)>,
    parameters: Parameters,
    alphabet: usize,
    order_bound: usize,
    levels: Vec<f64>,
    pub(crate) distribution: Vec<f64>,
    pub(crate) order: usize,
}

impl PpmDecay {
    pub(crate) fn new(alphabet: usize, order_bound: usize, parameters: Parameters) -> Self {
        assert!((1..=241).contains(&alphabet) && order_bound <= 4);
        Self {
            symbols: Vec::with_capacity(MAX_INTERVALS),
            times: Vec::with_capacity(MAX_INTERVALS),
            traces: Vec::with_capacity(MAX_INTERVALS * (order_bound + 1)),
            grams: HashMap::with_capacity(MAX_INTERVALS * (order_bound + 1)),
            parameters,
            alphabet,
            order_bound,
            levels: vec![0.0; (order_bound + 1) * alphabet],
            distribution: vec![0.0; alphabet],
            order: 0,
        }
    }

    pub(crate) fn release_unstarted_storage(&mut self) {
        assert!(self.symbols.is_empty());
        self.symbols.shrink_to_fit();
        self.times.shrink_to_fit();
        self.traces.shrink_to_fit();
        self.grams.shrink_to_fit();
    }

    pub(crate) fn clear(&mut self) {
        self.symbols.clear();
        self.times.clear();
        self.traces.clear();
        self.grams.clear();
        self.distribution.fill(0.0);
        self.order = 0;
    }

    fn weight(&self, gram: &[u16; 5], size: usize, time: f64) -> f64 {
        let p = &self.parameters;
        let mut total = 0.0;
        let mut index = self.grams.get(gram).map_or(EMPTY, |chain| chain.0);
        while index != EMPTY {
            let trace = &self.traces[index];
            let begin = trace.end + 1 - size;
            let mut expiry = self.times[begin] + p.buffer_time;
            let item_expiry = trace.end + p.buffer_items.saturating_sub(size).saturating_add(1);
            if let Some(&item_time) = self.times.get(item_expiry) {
                expiry = expiry.min(item_time);
            }
            let elapsed = time - expiry;
            total += if elapsed < 0.0 {
                p.buffer_weight
            } else if elapsed < p.stm_duration {
                p.stm_weight * (p.ltm_weight / p.stm_weight).powf(elapsed / p.stm_duration)
            } else {
                p.asymptote
                    + (p.ltm_weight - p.asymptote)
                        * 2.0f64.powf(-(elapsed - p.stm_duration) / p.half_life)
            };
            index = trace.next;
        }
        total
    }

    /// Retrieval updates scratch only. The caller supplies one paired noise block.
    pub(crate) fn predict(
        &mut self,
        time: f64,
        mut noise: impl FnMut() -> f64,
    ) -> Result<(), &'static str> {
        if !time.is_finite() || time < 0.0 || self.times.last().is_some_and(|&last| time < last) {
            return Err("prediction_before_history");
        }
        self.order = self.order_bound.min(self.symbols.len());
        while self.order > 0
            && time - self.times[self.times.len() - self.order] > self.parameters.buffer_time
        {
            self.order -= 1;
        }
        for size in (0..=self.order).rev() {
            let mut gram = [u16::MAX; 5];
            gram[..size].copy_from_slice(&self.symbols[self.symbols.len() - size..]);
            for candidate in 0..self.alphabet {
                gram[size] = candidate as u16;
                let increment = noise();
                if !increment.is_finite() || increment < 0.0 {
                    return Err("invalid_retrieval_noise");
                }
                self.levels[size * self.alphabet + candidate] =
                    self.weight(&gram, size + 1, time) + increment;
            }
        }
        self.distribution.fill(1.0 / (self.alphabet + 1) as f64);
        let expected_noise =
            self.alphabet as f64 * self.parameters.noise * (2.0 / std::f64::consts::PI).sqrt();
        for size in 0..=self.order {
            let counts = &self.levels[size * self.alphabet..(size + 1) * self.alphabet];
            let total: f64 = counts.iter().sum();
            let support = (total - expected_noise).max(0.0);
            let strength = if total > 0.0 {
                support / (support + 1.0)
            } else {
                0.0
            };
            for (lower, &count) in self.distribution.iter_mut().zip(counts) {
                *lower = if strength > 0.0 {
                    strength * count / total
                } else {
                    0.0
                } + (1.0 - strength) * *lower;
            }
        }
        let total: f64 = self.distribution.iter().sum();
        for value in &mut self.distribution {
            *value /= total;
        }
        if self
            .distribution
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err("invalid_sequence_distribution");
        }
        Ok(())
    }

    /// Learning never retrieves a distribution or consumes noise.
    pub(crate) fn learn(&mut self, symbol: u16, time: f64) -> Result<(), &'static str> {
        if symbol as usize >= self.alphabet
            || !time.is_finite()
            || time < 0.0
            || self.times.last().is_some_and(|&last| time <= last)
        {
            return Err("invalid_sequence_event");
        }
        if self.symbols.len() == MAX_INTERVALS {
            return Err("interval_resource_limit");
        }
        self.symbols.push(symbol);
        self.times.push(time);
        let end = self.symbols.len() - 1;
        for begin in end.saturating_sub(self.order_bound)..=end {
            if time - self.times[begin] >= self.parameters.buffer_time {
                continue;
            }
            let mut gram = [u16::MAX; 5];
            gram[..end - begin + 1].copy_from_slice(&self.symbols[begin..=end]);
            let index = self.traces.len();
            self.traces.push(Trace { end, next: EMPTY });
            if let Some(chain) = self.grams.get_mut(&gram) {
                self.traces[chain.1].next = index;
                chain.1 = index;
            } else {
                self.grams.insert(gram, (index, index));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn compare(case: &Value) -> usize {
        let p = &case["parameters"];
        let f = |key: &str| p[key].as_f64().unwrap();
        let parameters = Parameters {
            buffer_time: f("buffer_length_time"),
            buffer_items: f("buffer_length_items") as usize,
            buffer_weight: f("buffer_weight"),
            stm_weight: f("stm_weight"),
            stm_duration: f("stm_duration"),
            ltm_weight: f("ltm_weight"),
            half_life: f("ltm_half_life"),
            asymptote: f("ltm_asymptote"),
            noise: f("noise"),
        };
        let alphabet = case["alphabet_size"].as_u64().unwrap() as usize;
        let mut model = PpmDecay::new(
            alphabet,
            case["order_bound"].as_u64().unwrap() as usize,
            parameters,
        );
        let symbols = case["symbols"].as_array().unwrap();
        let times = case["times"].as_array().unwrap();
        let expected = case["expected"].as_array().unwrap();
        assert!(!symbols.is_empty());
        assert_eq!(symbols.len(), times.len());
        assert_eq!(symbols.len(), expected.len());
        for i in 0..symbols.len() {
            let time = times[i].as_f64().unwrap();
            let symbol = symbols[i].as_u64().unwrap() as u16;
            assert_eq!(time, expected[i]["time"].as_f64().unwrap());
            if let Some(label) = expected[i].get("symbol") {
                assert_eq!(label.as_u64().unwrap(), symbol as u64);
            }
            let mut noise = expected[i]["retrieval_noise"].as_array().unwrap().iter();
            let before = (
                model.symbols.clone(),
                model.times.clone(),
                model.traces.clone(),
                model.grams.clone(),
            );
            model
                .predict(time, || {
                    noise
                        .next()
                        .expect("complete noise block")
                        .as_f64()
                        .unwrap()
                })
                .unwrap();
            assert!(noise.next().is_none());
            assert_eq!(
                before,
                (
                    model.symbols.clone(),
                    model.times.clone(),
                    model.traces.clone(),
                    model.grams.clone()
                )
            );
            assert_eq!(model.order, expected[i]["order"].as_u64().unwrap() as usize);
            let distribution = expected[i]["distribution"].as_array().unwrap();
            assert_eq!(distribution.len(), alphabet);
            for (a, b) in model.distribution.iter().zip(distribution) {
                assert!((a - b.as_f64().unwrap()).abs() <= 2e-12);
            }
            assert!(
                (-model.distribution[symbol as usize].log2()
                    - expected[i]["loss_bits"].as_f64().unwrap())
                .abs()
                    <= 2e-12
            );
            model.learn(symbol, time).unwrap();
            assert_eq!(model.symbols.len(), i + 1);
            assert_eq!(model.times[i], time);
            assert_eq!(model.symbols[i], symbol);
        }
        if let Some(traces) = case.get("final_traces") {
            let traces = traces.as_array().unwrap();
            assert_eq!(model.grams.len(), traces.len());
            for trace in traces {
                let mut gram = [u16::MAX; 5];
                for (i, symbol) in trace["gram"].as_array().unwrap().iter().enumerate() {
                    gram[i] = symbol.as_u64().unwrap() as u16;
                }
                let mut index = model.grams[&gram].0;
                for end in trace["ends"].as_array().unwrap() {
                    assert_ne!(index, EMPTY);
                    assert_eq!(model.traces[index].end, end.as_u64().unwrap() as usize);
                    index = model.traces[index].next;
                }
                assert_eq!(index, EMPTY);
            }
        }
        symbols.len()
    }

    #[test]
    fn own_sequence_paired_upstream_fixture() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/ppm_decay_reference.json"
        ))
        .unwrap();
        let count: usize = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(compare)
            .sum();
        assert_eq!(count, 32);
    }

    #[test]
    fn own_sequence_adopted_reference_and_final_learning_state() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/own_sequence_reference.json"
        ))
        .unwrap();
        assert_eq!(compare(&fixture["cases"][0]), 32);
    }
}
