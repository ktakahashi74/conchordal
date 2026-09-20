//! Test-only capture of original finite episodes and exact long-return queries.

use super::*;
use std::io::Write;

pub(super) struct Audit {
    bus: u8,
    output: std::io::BufWriter<std::fs::File>,
    material: super::super::long_form::Material,
    graph_output: std::io::BufWriter<std::fs::File>,
    graph_checkpoint: usize,
    source_output: std::io::BufWriter<std::fs::File>,
    watched: Vec<Option<Watched>>,
}

struct Watched {
    query: u64,
    group: Handle,
    occurrence: u64,
    start: u64,
    last: Option<serde_json::Value>,
}

impl Audit {
    pub fn new(bus: u8, epoch: u64) -> Result<Option<Self>, &'static str> {
        let Some(path) = std::env::var_os("CONCHORDAL_I9_AUDIT") else {
            return Ok(None);
        };
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).map_err(|_| "create I9 capture directory")?;
        let file = std::fs::File::create(path.join(format!("bus{bus}-epoch{epoch}.jsonl")))
            .map_err(|_| "create I9 capture file")?;
        let material = super::super::long_form::Material::load();
        std::fs::write(
            path.join("material.json"),
            serde_json::to_vec_pretty(&material).unwrap(),
        )
        .map_err(|_| "write I9 capture material")?;
        Ok(Some(Self {
            bus,
            output: std::io::BufWriter::new(file),
            watched: std::iter::repeat_with(|| None)
                .take(material.returns.len())
                .collect(),
            material,
            graph_output: std::io::BufWriter::new(
                std::fs::File::create(path.join(format!("graph-bus{bus}-epoch{epoch}.jsonl")))
                    .map_err(|_| "create I9 graph capture")?,
            ),
            graph_checkpoint: 0,
            source_output: std::io::BufWriter::new(
                std::fs::File::create(path.join(format!("source-bus{bus}-epoch{epoch}.jsonl")))
                    .map_err(|_| "create I9 source ownership capture")?,
            ),
        }))
    }

    pub fn graph(
        &mut self,
        graph: &graph::Graph,
        cues: &super::super::section::cue::Stream,
        cut: u64,
        rate: u32,
    ) -> Result<(), &'static str> {
        for (index, watch) in self.watched.iter_mut().enumerate() {
            let Some(watch) = watch else { continue };
            if cut as f64 / f64::from(rate) > self.material.returns[index][1] + 2. {
                continue;
            }
            let state = cues.trace_prefix(watch.group, watch.occurrence, watch.start);
            if watch.last.as_ref() == Some(&state) {
                continue;
            }
            serde_json::to_writer(
                &mut self.source_output,
                &serde_json::json!({
                    "kind":"source_ownership", "checkpoint":index, "cut":cut, "sample_rate":rate,
                    "query_id":watch.query, "group":watch.group, "occurrence":watch.occurrence,
                    "start":watch.start, "state":state
                }),
            )
            .map_err(|_| "write I9 source ownership")?;
            writeln!(self.source_output).map_err(|_| "write I9 source ownership terminator")?;
            watch.last = Some(state);
        }
        self.source_output
            .flush()
            .map_err(|_| "flush I9 source ownership")?;
        if self.graph_checkpoint == self.material.returns.len()
            || (cut as f64 / f64::from(rate)) < self.material.returns[self.graph_checkpoint][1] + 2.
        {
            return Ok(());
        }
        let checkpoint = self.graph_checkpoint;
        serde_json::to_writer(
            &mut self.graph_output,
            &serde_json::json!({
                "kind":"snapshot", "checkpoint":checkpoint, "cut":cut, "sample_rate":rate,
                "summary":graph.snapshot()
            }),
        )
        .map_err(|_| "write I9 graph header")?;
        writeln!(self.graph_output).map_err(|_| "write I9 graph header terminator")?;
        for (source, edges) in graph.archived_nodes() {
            serde_json::to_writer(
                &mut self.graph_output,
                &serde_json::json!({
                    "kind":"node", "checkpoint":checkpoint, "source":source, "edges":edges
                }),
            )
            .map_err(|_| "write I9 graph node")?;
            writeln!(self.graph_output).map_err(|_| "write I9 graph node terminator")?;
        }
        for owner in graph.archived_prefixes() {
            serde_json::to_writer(
                &mut self.graph_output,
                &serde_json::json!({"kind":"prefix_owner", "checkpoint":checkpoint, "owner":owner}),
            )
            .map_err(|_| "write I9 graph prefix owner")?;
            writeln!(self.graph_output).map_err(|_| "write I9 graph owner terminator")?;
        }
        serde_json::to_writer(
            &mut self.graph_output,
            &serde_json::json!({
                "kind":"complete", "checkpoint":checkpoint
            }),
        )
        .map_err(|_| "write I9 graph checkpoint completion")?;
        writeln!(self.graph_output).map_err(|_| "write I9 graph completion terminator")?;
        self.graph_output
            .flush()
            .map_err(|_| "flush I9 graph capture")?;
        self.graph_checkpoint += 1;
        Ok(())
    }

    pub fn episode(
        &mut self,
        e: &memory::Episode,
        provenance: Option<super::super::section::commitment::Evidence>,
    ) -> Result<(), &'static str> {
        let value = serde_json::json!({"kind":"episode","id":e.identity.id,"generation":e.identity.generation,
            "epoch":e.epoch,"available_end":e.available_end,"first_observed_end":e.first_observed_end,
            "scales":e.scales,"knots":e.descriptor.knots,"provenance":provenance});
        serde_json::to_writer(&mut self.output, &value).map_err(|_| "write I9 episode")?;
        writeln!(self.output).map_err(|_| "write I9 episode terminator")
    }

    pub fn query(
        &mut self,
        dispatch: &query::Dispatch,
        episodes: &[memory::Episode],
        report: &memory::Report,
        rate: u32,
    ) -> Result<(), &'static str> {
        let end = dispatch.header.end;
        if !self
            .material
            .returns
            .iter()
            .any(|[begin, until]| *begin <= end && end < *until)
        {
            return Ok(());
        }
        for (index, [begin, until]) in self.material.returns.iter().enumerate() {
            if *begin + 2. <= end && end < *until && self.watched[index].is_none() {
                self.watched[index] = Some(Watched {
                    query: dispatch.header.query_id,
                    group: Handle {
                        bus: self.bus,
                        epoch: dispatch.header.epoch,
                        generation: dispatch.header.generation,
                    },
                    occurrence: dispatch.header.occurrence_id,
                    start: (dispatch.header.start * f64::from(rate)).round() as u64,
                    last: None,
                });
            }
        }
        let matches: Vec<_> = report
            .matches
            .iter()
            .map(|m| {
                serde_json::json!({"id":m.relation.identity.id,
            "cost":m.cost,"supported":m.relation.supported,"ambiguous":m.relation.ambiguous,
            "transformation":m.relation.transformation,"anchor":m.anchor})
            })
            .collect();
        let value = serde_json::json!({"kind":"query","query_id":dispatch.header.query_id,"occurrence_id":dispatch.header.occurrence_id,
            "epoch":dispatch.header.epoch,"start":dispatch.header.start,"end":end,
            "observed_end":dispatch.dispatched_at as f64/f64::from(rate),"scales":dispatch.header.scales,
            "knots":dispatch.descriptor.knots,"bank_ids":episodes.iter().map(|e|e.identity.id).collect::<Vec<_>>(),
            "cutoff_tie":report.cutoff_tie,"pruned_candidates":report.pruned_candidates,
            "pruned_ties":report.pruned_ties,"matches":matches});
        serde_json::to_writer(&mut self.output, &value).map_err(|_| "write I9 query")?;
        writeln!(self.output).map_err(|_| "write I9 query terminator")?;
        self.output.flush().map_err(|_| "flush I9 audit")
    }
}
