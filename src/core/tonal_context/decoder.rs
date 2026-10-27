//! Author-rule approximation: tabulated probe PP, retaining the observed L and G.
use super::*;

const PERIOD: usize = STEP * 2560;
const LOW_LOG2: f64 = 5.78135971352466 - 7. / 8.;
const GRID_CENTS: f64 = 1.;

fn probe(j: usize, hz: f64) -> f64 {
    if j >= 28800 {
        return 0.;
    }
    let u = j as f64 / 48000.;
    let e = if u < 0.01 {
        (1. - (std::f64::consts::PI * u / 0.01).cos()) / 2.
    } else if u >= 0.59 {
        (1. - (std::f64::consts::PI * (0.6 - u) / 0.01).cos()) / 2.
    } else {
        1.
    };
    0.1 * e * (2. * std::f64::consts::PI * hz * u).sin()
}

#[derive(Debug)]
struct Table {
    first: isize,
    rows: Vec<[f64; W]>,
}
impl Table {
    fn new(blank: &Model, hz: f64, previous: &[[f64; W]]) -> Self {
        let mut model = blank.clone();
        let first = model.pitch.frame as isize - previous.len() as isize - 147;
        let mut rows = previous.to_vec();
        for j in 0..40000 {
            if let Some(frame) = model.push(probe(j, hz)) {
                rows.push(frame.pp);
                if frame.index >= 147 + 61 {
                    break;
                }
            }
        }
        assert!(rows.len() > 70);
        Self { first, rows }
    }
    fn sample(&self, time: f64, lag: usize) -> Option<f64> {
        let x = time - self.first as f64;
        if x < 0. {
            return None;
        }
        let i = x.floor() as usize;
        let a = self.rows.get(i)?[lag];
        let b = self.rows.get(i + 1)?[lag];
        Some(a + (b - a) * x.fract())
    }
}

pub(crate) struct Decoder {
    tables: Vec<Table>,
}
impl std::fmt::Debug for Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decoder")
            .field("tables", &self.tables.len())
            .finish()
    }
}

impl Decoder {
    #[cfg(test)]
    pub(crate) fn unqueried_for_test() -> Arc<Self> {
        Arc::new(Self { tables: Vec::new() })
    }

    pub(crate) fn new() -> Self {
        let mut blank = Model::new(48000);
        let mut previous = Vec::new();
        for _ in 0..PERIOD {
            if let Some(frame) = blank.push(0.) {
                previous.push(frame.pp);
            }
        }
        let previous = &previous[previous.len() - 2..];
        let count = ((8000_f64.log2() - LOW_LOG2) * 1200. / GRID_CENTS).ceil() as usize + 1;
        let tables = (0..count)
            .map(|i| {
                let hz = (LOW_LOG2 + i as f64 * GRID_CENTS / 1200.).exp2();
                Table::new(&blank, hz, previous)
            })
            .collect();
        Self { tables }
    }
    #[cfg(test)]
    fn query(&self, model: &Model, hz: f64) -> Option<f64> {
        if !model.valid
            || model.sample_rate != 48000
            || model.pitch.frame == 0
            || !(55. * 2_f64.powf(-7. / 8.)..=8000.).contains(&hz)
        {
            return None;
        }
        self.query_memory(
            model.n0,
            model.pitch.frame,
            &model.pitch.local,
            &model.pitch.glob,
            hz,
        )
    }
    pub(super) fn query_memory(
        &self,
        n0: usize,
        frame: usize,
        local: &[f64; W],
        glob: &[f64; W],
        hz: f64,
    ) -> Option<f64> {
        if frame == 0 || !(55. * 2_f64.powf(-7. / 8.)..=8000.).contains(&hz) {
            return None;
        }
        let x = ((hz.log2() - LOW_LOG2) * 1200. / GRID_CENTS).max(0.);
        let lo = x.floor() as usize;
        let a = self.tables.get(lo)?;
        let b = self.tables.get(lo + 1)?;
        let mut local = *local;
        let mut glob = *glob;
        let al = 2_f64.powf(-1. / (FA / STEP as f64 * 0.1));
        let ag = 2_f64.powf(-1. / (FA / STEP as f64 * 1.5));
        let first = ((n0 + 9600) * 147).div_ceil(PERIOD);
        let end = ((n0 + 28800) * 147).div_ceil(PERIOD);
        let origin = n0 as f64 * 147. / PERIOD as f64;
        let mut sum = 0.;
        for k in frame..end {
            for i in 0..W {
                let pa = a.sample(k as f64 - origin, i)?;
                let pb = b.sample(k as f64 - origin, i)?;
                let pp = pa + (pb - pa) * x.fract();
                local[i] = al * local[i] + pp;
                glob[i] = ag * glob[i] + pp;
            }
            if k >= first {
                sum += pearson(&local, &glob)?;
            }
        }
        (end > first).then_some(sum / (end - first) as f64)
    }
}

#[test]
#[ignore = "TC1b-measure lock and retained runtime PCM/decisions required"]
fn actual_decisions() {
    replay(false);
}

#[test]
#[ignore = "TC1b-measure lock; saved paired queries and control keys required"]
fn interpolation_control() {
    replay(true);
}

#[cfg(test)]
fn replay(control: bool) {
    use std::collections::BTreeMap;
    use std::io::{BufRead, Read, Write};
    use std::time::Instant;
    let root = std::path::PathBuf::from(std::env::var("TC1B_INPUT").unwrap());
    let mut decisions: Vec<acquisition::Decision> =
        std::io::BufReader::new(std::fs::File::open(root.join("decisions.jsonl")).unwrap())
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();
    decisions.retain(|d| d.tick >= 106 * 48000 && d.dwell > 0.);
    let mut retained = BTreeMap::new();
    if control {
        let keys: Vec<(u64, u64)> =
            serde_json::from_slice(&std::fs::read(root.join("control-keys.json")).unwrap())
                .unwrap();
        decisions.retain(|d| keys.contains(&(d.id, d.tick)));
    }
    let retained_path = root.join(if control {
        "queries.jsonl"
    } else {
        "queries-input.jsonl"
    });
    if retained_path.exists() {
        for line in std::io::BufReader::new(std::fs::File::open(retained_path).unwrap()).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            retained.insert(
                (
                    row["decision"]["id"].as_u64().unwrap(),
                    row["decision"]["tick"].as_u64().unwrap(),
                ),
                row,
            );
        }
    }
    decisions.sort_by_key(|d| d.n0);
    assert!(
        !decisions.is_empty(),
        "the capture must include active plucks"
    );
    let start = Instant::now();
    let decoder = (!control).then(Decoder::new);
    let table_sec = start.elapsed().as_secs_f64();
    let mut output = std::fs::File::create(root.join(if control {
        "control.jsonl"
    } else {
        "queries.jsonl"
    }))
    .unwrap();
    if let Some(decoder) = &decoder {
        println!(
            "TABLE seconds={table_sec} nodes={} bytes={}",
            decoder.tables.len(),
            decoder
                .tables
                .iter()
                .map(|t| t.rows.len() * W * 8)
                .sum::<usize>()
        );
    }
    let mut input = std::io::BufReader::new(std::fs::File::open(root.join("habitat.bin")).unwrap());
    let mut mixed = Model::new(48000);
    let mut sources: BTreeMap<(u64, u32), Model> = BTreeMap::new();
    let mut next = 0;
    let mut floats = [0_u8; 512 * 4];
    let mut id_bytes = [0_u8; 8];
    let mut count_bytes = [0_u8; 4];
    let mut hop = 0_u64;
    while input.read_exact(&mut id_bytes).is_ok() {
        assert_eq!(u64::from_le_bytes(id_bytes), hop * 512);
        input.read_exact(&mut count_bytes).unwrap();
        let count = u32::from_le_bytes(count_bytes);
        input.read_exact(&mut floats).unwrap();
        let pcm: Vec<_> = floats
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let mut alive = Vec::new();
        for _ in 0..count {
            input.read_exact(&mut id_bytes).unwrap();
            input.read_exact(&mut count_bytes).unwrap();
            let key = (
                u64::from_le_bytes(id_bytes),
                u32::from_le_bytes(count_bytes),
            );
            alive.push(key);
            let state = sources.entry(key).or_insert_with(|| mixed.clone());
            input.read_exact(&mut floats).unwrap();
            for (&mix, own) in pcm.iter().zip(floats.as_chunks::<4>().0) {
                state.push((mix - f32::from_le_bytes(*own)) as f64);
            }
        }
        sources.retain(|key, _| alive.contains(key));
        for x in pcm {
            mixed.push(x as f64);
        }
        hop += 1;
        while next < decisions.len() && decisions[next].n0 <= mixed.n0 as u64 {
            let d = &decisions[next];
            assert_eq!(d.n0, mixed.n0 as u64);
            let state = if d.excluded {
                sources.get(&(d.id, d.generation))
            } else {
                Some(&mixed)
            };
            let Some(state) = state else {
                writeln!(output,"{}",serde_json::json!({"decision":d,"reason":"no_ownership_snapshot","queries":null})).unwrap();
                next += 1;
                continue;
            };
            let blank = control.then(|| {
                // Match every rational transport/PP phase without replaying a
                // long silent prefix. This control has no grid interpolation.
                let mut blank = Model::new(48000);
                for _ in 0..PERIOD + state.n0 % PERIOD {
                    blank.push(0.);
                }
                assert_eq!(state.pitch.frame % 147, blank.pitch.frame % 147);
                blank.pitch.local = state.pitch.local;
                blank.pitch.glob = state.pitch.glob;
                blank
            });
            let mut queries = BTreeMap::new();
            for hz in d
                .candidates
                .iter()
                .map(|&(p, _)| p.exp2())
                .chain([d.baseline.0.exp2(), d.hz])
            {
                queries.entry(hz.to_bits()).or_insert_with(|| {
                    if let Some(blank) = &blank {
                        let mut q = retained[&(d.id,d.tick)]["queries"][hz.to_bits().to_string()].clone();
                        assert_eq!(q["hz"].as_f64().unwrap(), hz as f64);
                        let start = Instant::now();
                        q["ideal"] = serde_json::json!(blank.query(hz as f64).unwrap().0);
                        q["ideal_us"] = serde_json::json!(start.elapsed().as_secs_f64()*1e6);
                        return q;
                    }
                    let start = Instant::now();
                    let old = retained.get(&(d.id,d.tick));
                    let exact = old.map_or_else(||state.query(hz as f64).unwrap().0,|r|r["queries"][hz.to_bits().to_string()]["exact"].as_f64().unwrap());
                    let exact_us = old.map_or_else(||Some(start.elapsed().as_secs_f64()*1e6), |_|None);
                    let start = Instant::now();
                    let approx = decoder.as_ref().unwrap().query(state,hz as f64).unwrap();
                    let approx_us = start.elapsed().as_secs_f64()*1e6;
                    serde_json::json!({"hz":hz,"exact":exact,"approx":approx,"exact_us":exact_us,"approx_us":approx_us})
                });
            }
            writeln!(output,"{}",serde_json::json!({"decision":d,"queries":queries,"table_seconds":table_sec,"grid_cents":GRID_CENTS})).unwrap();
            output.flush().unwrap();
            next += 1;
            println!(
                "DECISION {next}/{} tick={} n0={}",
                decisions.len(),
                d.tick,
                d.n0
            );
        }
        if next == decisions.len() {
            break;
        }
    }
    assert_eq!(next, decisions.len());
}
