pub(crate) mod analysis_worker;
pub(crate) mod auditory_frontend;
#[cfg(test)]
pub(crate) use auditory_frontend as auditory_shadow;
pub mod consonance_kernel;
pub(crate) mod db;
pub(crate) mod density;
pub(crate) mod erb;
pub(crate) mod fft;
pub(crate) mod float;
pub(crate) mod habituation;
pub(crate) mod harmonic_ratios;
pub(crate) mod harmonicity_kernel;
pub(crate) mod history_prediction;
pub mod landscape;
#[cfg(test)]
pub(crate) mod landscape_spectral;
pub mod log2space;
pub(crate) mod meter;
pub mod mode_pattern;
pub mod modulation;
pub mod nsgt_kernel;
pub(crate) mod nsgt_rt;
pub(crate) mod onset;
pub(crate) mod onset_habituation;
pub(crate) mod peak_extraction;
pub(crate) mod phase;
pub mod psycho_state;
pub(crate) mod roughness_kernel;
pub(crate) mod spectral_history;
pub(crate) mod stream;
pub(crate) mod temporal_expectation;
pub(crate) mod temporal_history;
pub mod timebase;
pub(crate) mod utils;

/// Runs `$body` once for each SIMD level this machine supports, binding the level's
/// name and token. Bodies wrap kernel calls in `$simd.vectorize(..)` so each level
/// compiles with its own target features.
#[cfg(test)]
macro_rules! for_each_simd_level {
    (|$name:ident, $simd:ident| $body:expr) => {{
        #[allow(unused_imports)]
        use fearless_simd::Simd as _;
        let level = fearless_simd::Level::new();
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some($simd) = level.as_sse2() {
                let $name = "sse2";
                $body;
            }
            if let Some($simd) = level.as_sse4_2() {
                let $name = "sse4.2";
                $body;
            }
            if let Some($simd) = level.as_avx2() {
                let $name = "avx2";
                $body;
            }
            if let Some($simd) = level.as_avx512() {
                let $name = "avx512";
                $body;
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some($simd) = level.as_neon() {
            let $name = "neon";
            $body;
        }
    }};
}
#[cfg(test)]
pub(crate) use for_each_simd_level;
pub(crate) mod tonal_context;
