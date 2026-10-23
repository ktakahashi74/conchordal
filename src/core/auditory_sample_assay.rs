//! Fixed SF-1 habitat PCM: frequency correspondence only, never a mixture oracle.
use super::*;
use crate::core::landscape_spectral::SpectralFrontEnd;
use crate::core::peak_extraction::{
    PeakExtractConfig, PeakScratch, extract_peaks_density_with_grid_reuse,
};
use std::io::{BufRead, Read, Write};

fn mass_peaks(space: &Log2Space, density: &[f32], du: &[f32]) -> Vec<[f32; 2]> {
    space.assert_scan_len_named(density, "sample_density_scan");
    space.assert_scan_len_named(du, "sample_du_scan");
    let mass: Vec<_> = density.iter().zip(du).map(|(d, w)| d * w).collect();
    (0..mass.len())
        .filter(|&i| {
            mass[i] > 0.0
                && (i == 0 || mass[i] >= mass[i - 1])
                && (i + 1 == mass.len() || mass[i] > mass[i + 1])
        })
        .map(|i| [space.centers_hz[i], mass[i]])
        .collect()
}

#[test]
#[ignore = "AUD-M1 immutable 17_flora and drone90 PCM correspondence table"]
fn aud_m1_sample_table() {
    use sha2::Digest;
    let source = PathBuf::from(std::env::var("AUD_M1_SAMPLE_ROOT").unwrap());
    let dir = output_dir();
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let space = kernel.space().clone();
    let (erb, du) = erb_grid(&space);
    let mut summaries = Vec::new();
    for scene in ["17_flora", "drone90"] {
        let input = source.join(scene);
        let mut native =
            std::io::BufReader::new(std::fs::File::open(input.join("native.jsonl")).unwrap());
        let mut native_hash = sha2::Sha256::new();
        let mut line = String::new();
        let mut flora = Vec::new();
        while native.read_line(&mut line).unwrap() > 0 {
            native_hash.update(line.as_bytes());
            let row: Value = serde_json::from_str(&line).unwrap();
            if row["type"] == "flora" {
                flora.push((
                    row["time_sec"].as_f64().unwrap(),
                    row["flora_id"].clone(),
                    row["rendered_partials"].clone(),
                ));
            }
            line.clear();
        }
        let mut pcm_file =
            std::io::BufReader::new(std::fs::File::open(input.join("habitat.f32le")).unwrap());
        let byte_length = pcm_file.get_ref().metadata().unwrap().len() as usize;
        assert_eq!(byte_length % (4 * hop), 0);
        let frames = byte_length / (4 * hop);
        let mut bytes = vec![0; 4 * hop];
        let mut pcm = vec![0.0; hop];
        let mut pcm_hash = sha2::Sha256::new();
        let mut rt = kernel.clone();
        let mut old = SpectralFrontEnd::new(space.clone());
        let mut selected =
            ShadowFrontEnd::new(space.clone(), params.fs, 0.25, false, kernel.nfft())
                .with_statistics(hop, true);
        let mut density = vec![0.0; space.n_bins()];
        let mut peak_scratch = PeakScratch::new(space.n_bins());
        let mut writer = std::io::BufWriter::new(
            std::fs::File::create(dir.join(format!("{scene}-rows.jsonl"))).unwrap(),
        );
        let mut occurrences = 0_usize;
        let mut hits = [0_usize; 3];
        let mut snapshots = 0;
        let mut unmatched_times = 0;
        for frame in 0..frames {
            pcm_file.read_exact(&mut bytes).unwrap();
            pcm_hash.update(&bytes);
            for (x, b) in pcm.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *x = f32::from_le_bytes(*b);
            }
            let power = rt.process_hop(&pcm);
            let old_view = old.process_nsgt_power(power, hop as f32 / params.fs, &params);
            selected.process(&pcm, &params);
            if frame * hop / 48000 == (frame + 1) * hop / 48000 {
                continue;
            }
            snapshots += 1;
            for ((d, &p), &w) in density.iter_mut().zip(power).zip(&du) {
                *d = p / w;
            }
            let peaks = extract_peaks_density_with_grid_reuse(
                &density,
                &erb,
                &du,
                &PeakExtractConfig::nsgt_default(),
                &mut peak_scratch,
            );
            let old_raw: Vec<_> = peaks
                .iter()
                .map(|p| [space.centers_hz[p.bin_idx], p.mass])
                .collect();
            let old_mass = mass_peaks(&space, old_view.subjective_intensity, &du);
            let new_mass = mass_peaks(&space, &selected.density_scan, &du);
            let end = (frame + 1) as f64 * hop as f64 / f64::from(params.fs);
            let nearest = flora.iter().enumerate().min_by(|(ia, a), (ib, b)| {
                (a.0 - end)
                    .abs()
                    .total_cmp(&(b.0 - end).abs())
                    .then(ia.cmp(ib))
            });
            let mut details = Vec::new();
            let mut time = Value::Null;
            let mut id = Value::Null;
            let mut matched = false;
            if let Some((_, entry)) = nearest
                && (entry.0 - end).abs() <= hop as f64 / f64::from(params.fs) + 1e-4
            {
                matched = true;
                time = json!(entry.0);
                id = entry.1.clone();
                for partial in entry.2.as_array().unwrap() {
                    let f = partial[0].as_f64().unwrap();
                    let power = partial[1].as_f64().unwrap();
                    if !partial[2].as_bool().unwrap() || power <= 0.0 {
                        continue;
                    }
                    let hit = [&old_raw, &old_mass, &new_mass].map(|peaks| {
                        peaks
                            .iter()
                            .any(|p| (f / f64::from(p[0])).log2().abs() <= 1.0 / 96.0)
                    });
                    occurrences += 1;
                    for i in 0..3 {
                        hits[i] += usize::from(hit[i]);
                    }
                    details.push(
                        json!({"frequency_hz":f,"power":power,"old_raw_old_mass_new_mass":hit}),
                    );
                }
            }
            if !matched {
                unmatched_times += 1;
            }
            writeln!(writer,"{}",json!({"frame":frame,"end_sec":end,"native_time":time,"flora_id":id,"time_matched":matched,
                "rendered_partials":details,"old_raw_peaks":old_raw,"current_mass_peaks":old_mass,"selected_mass_peaks":new_mass})).unwrap();
        }
        writer.flush().unwrap();
        summaries.push(json!({"scene":scene,"frames":frames,"fs":params.fs,"hop":hop,"snapshots":snapshots,"unmatched_times":unmatched_times,
            "flora_occurrences":occurrences,"old_raw_matched":hits[0],"current_mass_matched":hits[1],"selected_mass_matched":hits[2],
            "pcm_path":input.join("habitat.f32le"),"pcm_bytes":byte_length,"pcm_sha256":pcm_hash.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "native_path":input.join("native.jsonl"),"native_sha256":native_hash.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),
            "matching":"SF-1: closest Flora row to the second crossing, first row on time ties; time tolerance hop/fs+1e-4; positive tonal rendered partials; frequency tolerance 12.5 cent",
            "mass_peak_rule":"positive density*du local maxima, rightmost bin on a plateau; no new relative threshold or cap",
            "scope":"frequency correspondence on immutable old-main PCM; cannot attribute a mixed peak to Flora rather than a coincident Voice"}));
    }
    std::fs::write(
        dir.join("summary.json"),
        serde_json::to_vec_pretty(&summaries).unwrap(),
    )
    .unwrap();
}
