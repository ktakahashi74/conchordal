use conchordal::{Limiter, StepEngine, StepStatus};
use wasm_bindgen::prelude::*;

/// The host owns scheduling; each call produces exactly one mono hop.
#[wasm_bindgen]
pub struct BrowserEngine {
    engine: StepEngine,
    presentation: Vec<f32>,
    limiter: Limiter,
}

#[wasm_bindgen]
impl BrowserEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(script: &str, config: &str, seed: u32, sample_rate: u32) -> Result<Self, JsValue> {
        let engine = StepEngine::new(script, config, Some(u64::from(seed)), sample_rate)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let presentation = vec![0.0; engine.hop_len()];
        let limiter = Limiter::new(engine.limiter_mode(), sample_rate, 1)
            .with_input_gain(engine.output_gain());
        Ok(Self {
            engine,
            presentation,
            limiter,
        })
    }

    pub fn hop_len(&self) -> usize {
        self.engine.hop_len()
    }

    pub fn step(&mut self) -> Result<Vec<f32>, JsValue> {
        if let StepStatus::Error(message) = self.engine.step(&mut self.presentation) {
            return Err(JsValue::from_str(&message));
        }
        self.limiter.process_interleaved(&mut self.presentation, 1);
        Ok(self.presentation.clone())
    }

    pub fn snapshot(&mut self) -> Result<Vec<u8>, JsValue> {
        self.engine
            .ui_frame()
            .ok_or_else(|| JsValue::from_str("GUI frame unavailable"))?
            .web_snapshot()
            .encode()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn request_end(&mut self) {
        self.engine.request_end();
    }
    pub fn finished(&self) -> bool {
        self.engine.finished()
    }
}
