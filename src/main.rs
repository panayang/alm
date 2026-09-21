//! Command line driver.
//!
//!   cargo run --release -- gencheck            data self-tests only
//!   cargo run --release -- quick               short run, everything
//!   cargo run --release -- full                the whole suite
//!
//! Options: --ticks N  --seed S  --out FILE

use std::fs;

use alm::experiments;
use alm::gen::GenConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("screen");

    let mut ticks: usize = match cmd {
        "full" => 600_000,
        "screen" => 90_000,
        _ => 90_000,
    };
    let mut seed: u64 = 0x5EED_1234;
    let mut out: Option<String> = None;
    let (mut shard, mut shards) = (0usize, 1usize);
    let mut trace: Option<String> = None;
    let mut label = String::from("trace");
    let mut limit: usize = 0;
    let mut gran = String::from("line");
    let mut bankses: Vec<usize> = vec![4096, 8192, 16384, 32768];
    let mut queries: usize = 2000;
    let mut sample: usize = 16;
    let mut data: Option<String> = None;
    let mut bins: usize = 16;
    let mut max_given: usize = 3;
    let mut min_info: f64 = 0.05;
    let mut budgets: Vec<usize> = vec![64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20];
    let mut widths: Vec<usize> = vec![32, 64, 128, 256, 512, 1024, 2048];
    // Overrides Config::local's step size, so a runner can be asked to choose
    // it on real data rather than on the two diagnostic streams.
    let mut eta: Option<f32> = None;
    let mut verify_gate = false;
    let mut held: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--ticks" => {
                ticks = args[i + 1].parse().expect("--ticks wants a number");
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().expect("--seed wants a number");
                i += 2;
            }
            "--out" => {
                out = Some(args[i + 1].clone());
                i += 2;
            }
            "--shard" => {
                let v = &args[i + 1];
                let (a, b) = v.split_once('/').expect("--shard wants i/n");
                shard = a.parse().expect("shard index");
                shards = b.parse().expect("shard count");
                i += 2;
            }
            "--trace" => {
                trace = Some(args[i + 1].clone());
                i += 2;
            }
            "--label" => {
                label = args[i + 1].clone();
                i += 2;
            }
            "--banks" => {
                bankses = args[i + 1].split(',').map(|x| x.parse().expect("banks")).collect();
                i += 2;
            }
            "--data" => {
                data = Some(args[i + 1].clone());
                i += 2;
            }
            "--bins" => {
                bins = args[i + 1].parse().expect("--bins wants a number");
                i += 2;
            }
            "--max-given" => {
                max_given = args[i + 1].parse().expect("--max-given wants a number");
                i += 2;
            }
            "--min-info" => {
                min_info = args[i + 1].parse().expect("--min-info wants a number");
                i += 2;
            }
            "--sample" => {
                sample = args[i + 1].parse().expect("--sample wants a number");
                i += 2;
            }
            "--budgets" => {
                budgets = args[i + 1]
                    .split(',')
                    .map(|x| x.trim_end_matches(['K', 'M']).parse::<usize>().expect("budget")
                        * if x.ends_with('M') { 1 << 20 } else { 1 << 10 })
                    .collect();
                i += 2;
            }
            "--queries" => {
                queries = args[i + 1].parse().expect("--queries wants a number");
                i += 2;
            }
            "--widths" => {
                widths = args[i + 1].split(',').map(|x| x.parse().expect("width")).collect();
                i += 2;
            }
            "--gran" => {
                gran = args[i + 1].clone();
                i += 2;
            }
            "--held" => {
                held = Some(args[i + 1].clone());
                i += 2;
            }
            "--verify-gate" => {
                verify_gate = true;
                i += 1;
            }
            "--eta" => {
                eta = Some(args[i + 1].parse().expect("--eta wants a number"));
                i += 2;
            }
            "--limit" => {
                limit = args[i + 1].parse().expect("--limit wants a number");
                i += 2;
            }
            other => panic!("unknown argument {}", other),
        }
    }

    match cmd {
        // Properties of a candidate stream, before any ingest is written for
        // it. Runs nothing of the model.
        "scan" => {
            let t = trace.expect("scan wants --trace PATH (or --trace - for stdin)");
            alm::scan::run(&t, &label, limit, &gran);
        }
        // Does the timing of a clinical record carry its outcome, on its own?
        // The gate for this architecture's temporal commitment. Runs no model.
        "clinical" => {
            let d = data.expect("clinical wants --data DIR");
            let o = out.clone().expect("clinical wants --out OUTCOMES.txt");
            alm::clinical::run(&d, &o, &label, bins);
        }
        // The label-free question: do the event stream and the gap stream
        // inform each other? Still no model.
        "clinical-next" => {
            let d = data.expect("clinical-next wants --data DIR");
            alm::clinical::next_event(&d, &label, max_given);
        }
        // What the N+1-th thing costs given the N already stored, split by
        // whether a counter could have helped at all.
        "acquire" => {
            let dd = data.clone().expect("acquire wants --data DIR");
            alm::acquire::run(&dd, &label, widths[0], limit, seed, max_given, bins, bankses[0], eta, verify_gate);
        }
        // The model's state read as a patient, scored on the benchmark's own
        // task against its own baselines.
        "patient" => {
            let dd = data.clone().expect("patient wants --data DIR");
            let o = out.clone().expect("patient wants --out OUTCOMES.txt");
            alm::patient::run(
                &dd,
                &o,
                &label,
                widths[0],
                limit,
                seed,
                max_given,
                bins,
                gran == "paramval",
                trace.as_deref(),
            );
        }
        // The tick loop on a real irregular stream. The first time the model
        // runs on a source it was not written for.
        "physio" => {
            let dd = data.expect("physio wants --data DIR");
            alm::physio::run(&dd, &label, &widths, limit, seed, max_given, &gran);
        }
        // Does the readout converge or hover? Two streams with known answers,
        // priced by decile so a curve that turns around cannot read as a
        // plateau. This is what chose `eta` and retired `row_norm_cap`.
        "stepsize" => {
            alm::stepsize::run(if limit > 0 { limit } else { 120_000 });
        }
        // Can the readout answer from a part when the whole is new? The one
        // cell PhysioNet still loses, with a known answer of zero bits.
        "compose" => {
            alm::compose::run(if limit > 0 { limit } else { 40_000 }, max_given.min(8));
        }
        // Sweep a probe's hyperparameters against a representation that was
        // produced once, so the sweep costs minutes instead of hours.
        "probe" => {
            let f = data.expect("probe wants --data DUMPFILE");
            alm::probe::run(&f);
        }
        // Is there anything left in this data that we have not taken? Replay
        // three quarters of the patients, read only the quarter held back.
        "epochs" => {
            let dd = data.clone().expect("epochs wants --data DIR");
            match held.clone() {
                Some(h) => alm::epochs::run_held(
                    &dd, &h, widths[0], limit, seed, max_given, bins, bankses[0],
                ),
                None => alm::epochs::run(&dd, widths[0], limit, seed, max_given, bins, bankses[0]),
            }
        }
        // Does consolidation cost plasticity? A change-point stream whose right
        // charge is zero on both sides, and three rules for how far a row still
        // moves.
        "plastic" => {
            alm::plastic::run(if limit > 0 { limit } else { 200_000 });
        }
        // Is the readout calibrated on rare tokens? The right charge on an
        // i.i.d. stream is -log2 p exactly, so the excess by rarity bucket names
        // whichever mechanism under-weights what it has seen rarely.
        "rarity" => {
            alm::rarity::run(if limit > 0 { limit } else { 120_000 });
        }
        // What a concept-drift stream is made of, and whether it has any
        // inter-arrival structure for us to spend. Runs no model.
        "stream" => {
            let f = data.expect("stream wants --data FILE");
            alm::stream::run(&f, &label, bins, if limit > 0 { limit } else { 1000 });
        }
        // What a record table costs to answer from every direction at once.
        // Runs no model; this is a property of the table.
        "partial" => {
            let f = data.expect("partial wants --data FILE");
            alm::partial::run(&f, &label, bins, max_given, &widths, min_info);
        }
        // Same metadata budget, spent two ways: our superposition against an
        // exact table with LRU. The only comparison this domain cares about.
        "budget" => {
            let t = trace.expect("budget wants --trace PATH (or - for stdin)");
            alm::budget::run(&t, &label, &gran, &budgets, &widths, limit, sample, seed);
        }
        // The same chain, but held in our own banked superposition instead of
        // an exact table. Sweeps the width against the capacity law.
        "chainmem" => {
            let t = trace.expect("chainmem wants --trace PATH (or - for stdin)");
            alm::chainmem::run(&t, &label, &gran, &widths, &bankses, limit, queries, seed);
        }
        "gencheck" => {
            // Both rungs of the load sweep, so the manipulation each family
            // actually received is visible side by side.
            for domains in [12usize, 36] {
                let mut g = GenConfig::fast();
                g.seed = seed ^ 0xA11CE;
                g.domains = domains;
                let t = ticks * domains / 12;
                println!("== domains={} ticks={} ==", domains, t);
                let stream = experiments::build_stream(&g, t, 20, true);
                println!("stream ok: {} ticks, {} episodes", stream.len(), stream.episodes.len());
            }
        }
        // The baseline alone, so a reporting fix does not cost a full suite run.
        "baseline" => {
            // Same source the screening suite uses, or the two numbers are
            // not about the same stream.
            let mut g = GenConfig::fast();
            g.seed = seed ^ 0xA11CE;
            let stream = experiments::build_stream(&g, ticks, 20, false);
            for (name, skip) in [("like-for-like", false), ("silence removed (UPPER BOUND)", true)] {
                let r = alm::baseline::run(&stream, 4, skip);
                println!(
                    "[baseline {}] charged events: {:.3} bits/ev, acc {:.3} |                      second-order {:.3} bits, acc {:.3} | product {:.3} bits, acc {:.3}                      | per-symbol incl. silence {:.3} (different denominator)",
                    name,
                    r.charged.mean(), r.charged.accuracy(),
                    r.second.mean(), r.second.accuracy(),
                    r.product.mean(), r.product.accuracy(),
                    r.ppm.bits_per_event()
                );
            }
        }
        "closeout" => {
            let suite = experiments::closeout(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "unbindtest" => {
            let suite = experiments::unbindtest(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "loadcurve" => {
            let suite = experiments::loadcurve(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "minimal" => {
            let suite = experiments::minimal(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "worth" => {
            let suite = experiments::worth(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "grid" => {
            let suite = experiments::grid(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "negatives" => {
            let suite = experiments::negatives(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "bindprobe" => {
            let suite = experiments::bindprobe(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "binddecay" => {
            let suite = experiments::binddecay(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "freeze2" => {
            let suite = experiments::freeze2(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "depth" => {
            let suite = experiments::depth(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "seeds" => {
            let suite = experiments::seeds(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "route" => {
            let suite = experiments::route(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "mechanism" => {
            let suite = experiments::mechanism(ticks, seed, shard, shards);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "width" => {
            let suite = experiments::width(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "load" => {
            let suite = experiments::load_sweep(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "scale" => {
            let suite = experiments::scale(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "capacity" => {
            let suite = experiments::capacity(ticks, seed);
            println!();
            for line in suite.summary.iter() {
                println!("{}", line);
            }
        }
        "screen" | "full" => {
            let suite = experiments::screen(ticks, seed, cmd == "full");
            println!();
            println!("== summary ====================================================");
            for line in suite.summary.iter() {
                println!("{}", line);
            }
            if let Some(path) = out {
                fs::write(&path, &suite.csv).expect("could not write csv");
                println!("\ncsv written to {}", path);
            }
        }
        other => panic!("unknown command {} (try gencheck, baseline, capacity, screen, full)", other),
    }
}
