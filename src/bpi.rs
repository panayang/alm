//! Questions about a case in progress, on a real process log.
//!
//! **Open-loop instrument.** The world's next event here is fixed in advance
//! and ignores what the model says, so this measures memory and prediction,
//! not response. It is kept as a record; it is not the test this design is
//! for (see the top of `lib.rs`).
//!
//! PhysioNet taught what this architecture is not for. In-hospital death is a
//! function of how much and how bad, summed over a stay -- a bag of counts gets
//! 0.80 and adding order adds nothing -- and the architecture's state is a
//! recency-weighted memory, so the present state holds 0.71 of it. That label
//! is a prediction: it fixes an answer at one position. What the architecture
//! did well on, in synthetic form, was a different kind of task: a question
//! about the history so far, asked at any moment, answered by the world, many
//! times per sequence.
//!
//! A business process log is that kind of source. BPI Challenge 2012: a Dutch
//! financial institute's loan applications, 13087 cases, 262200 events, each
//! event an activity and a lifecycle stage with a timestamp. At any point in a
//! case one can ask whether an offer has gone out, whether the latest offer
//! event was a cancellation, whether a work item is still open -- and the log
//! itself fixes the answer exactly, with no label noise.
//!
//! This file does two things. It reads the log into the stream shape the rest
//! of the project uses (gap, token). And it screens the questions before any
//! model sees them: for each, how much of the answer is available to a judge
//! that sees only the last event, the last two, or the counts with order thrown
//! away. A question those judges already answer is not one on which a state
//! can show anything, however good it is; the screen is what PhysioNet should
//! have been put through first.

use std::collections::HashMap;

pub struct Log {
    /// Each case: (seconds since the previous event, token).
    pub cases: Vec<Vec<(u32, u32)>>,
    /// Token id to "activity|lifecycle".
    pub names: Vec<String>,
}

fn attr<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("key=\"{}\" value=\"", key);
    let i = line.find(&k)? + k.len();
    let j = line[i..].find('"')? + i;
    Some(&line[i..j])
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// "2011-10-01T00:38:44.546+02:00" to seconds since the epoch, UTC.
fn parse_ts(s: &str) -> Option<i64> {
    let y: i64 = s.get(0..4)?.parse().ok()?;
    let mo: i64 = s.get(5..7)?.parse().ok()?;
    let d: i64 = s.get(8..10)?.parse().ok()?;
    let h: i64 = s.get(11..13)?.parse().ok()?;
    let mi: i64 = s.get(14..16)?.parse().ok()?;
    let se: i64 = s.get(17..19)?.parse().ok()?;
    let tz = s.rfind(|c| c == '+' || c == '-').filter(|&p| p > 19)?;
    let sign = if &s[tz..tz + 1] == "-" { -1 } else { 1 };
    let th: i64 = s.get(tz + 1..tz + 3)?.parse().ok()?;
    let tm: i64 = s.get(tz + 4..tz + 6)?.parse().ok()?;
    let local = days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se;
    Some(local - sign * (th * 3600 + tm * 60))
}

pub fn load(path: &str) -> Log {
    let body = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    let mut ids: HashMap<String, u32> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut cases = Vec::new();
    let mut cur: Vec<(i64, u32)> = Vec::new();
    let (mut in_event, mut name, mut life, mut ts) = (false, None, None, None);
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("<trace>") {
            cur.clear();
        } else if t.starts_with("</trace>") {
            cur.sort_by_key(|x| x.0);
            let mut out = Vec::with_capacity(cur.len());
            let mut last = cur.first().map(|x| x.0).unwrap_or(0);
            for &(t, tok) in cur.iter() {
                out.push(((t - last).max(0) as u32, tok));
                last = t;
            }
            if !out.is_empty() {
                cases.push(out);
            }
        } else if t.starts_with("<event>") {
            in_event = true;
            name = None;
            life = None;
            ts = None;
        } else if t.starts_with("</event>") {
            in_event = false;
            if let (Some(n), Some(l), Some(s)) = (name.take(), life.take(), ts.take()) {
                let key = format!("{}|{}", n, l);
                let n_ids = ids.len() as u32;
                let id = *ids.entry(key.clone()).or_insert_with(|| {
                    names.push(key);
                    n_ids
                });
                cur.push((s, id));
            }
        } else if in_event {
            if let Some(v) = attr(t, "concept:name") {
                name = Some(v.to_string());
            } else if let Some(v) = attr(t, "lifecycle:transition") {
                life = Some(v.to_string());
            } else if let Some(v) = attr(t, "time:timestamp") {
                ts = parse_ts(v);
            }
        }
    }
    Log { cases, names }
}

/// A question about a case in progress, answered from its prefix alone: the
/// events so far, and the time of each since the case began, in seconds.
pub struct Question {
    pub name: &'static str,
    /// None when the question does not apply yet at this prefix.
    pub answer: fn(&[&str], &[i64]) -> Option<bool>,
}

fn act(n: &str) -> &str {
    n.split('|').next().unwrap_or(n)
}
fn life(n: &str) -> &str {
    n.split('|').nth(1).unwrap_or("")
}

pub fn questions() -> Vec<Question> {
    vec![
        Question {
            name: "an offer has been sent",
            answer: |p, _| Some(p.iter().any(|n| act(n) == "O_SENT")),
        },
        Question {
            name: "more than one offer created",
            answer: |p, _| Some(p.iter().filter(|n| act(n) == "O_CREATED").count() >= 2),
        },
        Question {
            name: "latest offer event is a cancellation",
            answer: |p, _| {
                p.iter()
                    .rev()
                    .find(|n| act(n).starts_with("O_"))
                    .map(|n| act(n) == "O_CANCELLED")
            },
        },
        Question {
            name: "offer follow-up call still open",
            answer: |p, _| {
                p.iter()
                    .rev()
                    .find(|n| act(n) == "W_Nabellen offertes")
                    .map(|n| life(n) != "COMPLETE")
            },
        },
        Question {
            name: "an offer re-created after a cancellation",
            answer: |p, _| {
                let c = p.iter().position(|n| act(n) == "O_CANCELLED")?;
                Some(p[c + 1..].iter().any(|n| act(n) == "O_CREATED"))
            },
        },
        // Time. Answerable only by remembering when a particular earlier event
        // happened and accumulating the silences since -- which is what this
        // architecture's handling of silence is for, and what no judge that
        // sees tokens without time can do.
        Question {
            name: "over 7 days since the offer was sent",
            answer: |p, t| {
                let i = p.iter().position(|n| act(n) == "O_SENT")?;
                Some(t[t.len() - 1] - t[i] > 7 * 86400)
            },
        },
        Question {
            name: "sent offer unanswered over 14 days",
            answer: |p, t| {
                let i = p.iter().position(|n| act(n) == "O_SENT")?;
                if p[i..].iter().any(|n| act(n) == "O_SENT_BACK") {
                    return Some(false);
                }
                Some(t[t.len() - 1] - t[i] > 14 * 86400)
            },
        },
        Question {
            name: "case open over 10 days",
            answer: |_, t| Some(t[t.len() - 1] > 10 * 86400),
        },
    ]
}

/// Order of magnitude of a duration: 0 under a minute, then one step per
/// factor of four. What a time-aware judge is allowed to see of a duration.
fn mag(sec: i64) -> u64 {
    let mut s = (sec.max(0) / 60) as u64;
    let mut k = 0u64;
    while s > 0 {
        s /= 4;
        k += 1;
    }
    k
}

/// Majority-lookup judge: fit key -> majority answer on the first half of the
/// cases, score on the second; an unseen key falls back to the overall
/// majority. The key is whatever the judge is allowed to see.
fn lookup_accuracy(rows: &[(u64, bool, bool)]) -> f64 {
    // rows: (key, answer, in_test)
    let mut tab: HashMap<u64, (u32, u32)> = HashMap::new();
    let (mut yes, mut all) = (0u32, 0u32);
    for (k, a, test) in rows.iter() {
        if !*test {
            let e = tab.entry(*k).or_insert((0, 0));
            e.1 += 1;
            all += 1;
            if *a {
                e.0 += 1;
                yes += 1;
            }
        }
    }
    let global = yes * 2 >= all;
    let (mut hit, mut n) = (0u64, 0u64);
    for (k, a, test) in rows.iter() {
        if *test {
            let guess = match tab.get(k) {
                Some(&(y, t)) => y * 2 >= t,
                None => global,
            };
            n += 1;
            if guess == *a {
                hit += 1;
            }
        }
    }
    hit as f64 / n.max(1) as f64
}

fn hash(parts: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &p in parts {
        h = (h ^ p).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn screen(path: &str) {
    let log = load(path);
    let events: usize = log.cases.iter().map(|c| c.len()).sum();
    println!("BPI Challenge 2012: {} cases, {} events, {} tokens (activity x lifecycle)", log.cases.len(), events, log.names.len());
    let mut gaps: Vec<u32> = log.cases.iter().flat_map(|c| c.iter().skip(1).map(|x| x.0)).collect();
    gaps.sort_unstable();
    let q = |p: f64| gaps[((gaps.len() - 1) as f64 * p) as usize];
    println!(
        "  gaps between events (s): p10 {}  p50 {}  p90 {}  max {}",
        q(0.1),
        q(0.5),
        q(0.9),
        gaps[gaps.len() - 1]
    );
    println!("\n  screening: can a judge without the full history already answer?");
    println!("  first half of cases to fit, second half to score; every prefix of every case is a question.\n");
    println!(
        "{:>40} {:>8} {:>7} {:>8} {:>8} {:>8} {:>10} {:>11} {:>6}",
        "question", "n", "P(yes)", "last 1", "last 2", "counts", "last+gap", "counts+age", "full"
    );
    let half = log.cases.len() / 2;
    for qu in questions().iter() {
        let mut r1: Vec<(u64, bool, bool)> = Vec::new();
        let mut r2: Vec<(u64, bool, bool)> = Vec::new();
        let mut rb: Vec<(u64, bool, bool)> = Vec::new();
        let mut rg: Vec<(u64, bool, bool)> = Vec::new();
        let mut ra: Vec<(u64, bool, bool)> = Vec::new();
        let mut yes = 0usize;
        for (ci, c) in log.cases.iter().enumerate() {
            let test = ci >= half;
            let names: Vec<&str> = c.iter().map(|x| log.names[x.1 as usize].as_str()).collect();
            let mut times: Vec<i64> = Vec::with_capacity(c.len());
            let mut acc = 0i64;
            for (k, x) in c.iter().enumerate() {
                if k > 0 {
                    acc += x.0 as i64;
                }
                times.push(acc);
            }
            let mut counts = vec![0u64; log.names.len()];
            for k in 0..c.len() {
                counts[c[k].1 as usize] = (counts[c[k].1 as usize] + 1).min(3);
                let Some(a) = (qu.answer)(&names[..=k], &times[..=k]) else { continue };
                if a {
                    yes += 1;
                }
                let last = c[k].1 as u64;
                let prev = if k > 0 { c[k - 1].1 as u64 } else { u64::MAX };
                let gap = if k > 0 { c[k].0 as i64 } else { 0 };
                r1.push((last, a, test));
                r2.push((hash(&[prev, last]), a, test));
                rb.push((hash(&counts), a, test));
                rg.push((hash(&[last, mag(gap)]), a, test));
                let mut ck = counts.clone();
                ck.push(mag(times[k]));
                ra.push((hash(&ck), a, test));
            }
        }
        let n = r1.len();
        println!(
            "{:>40} {:>8} {:>7.3} {:>8.3} {:>8.3} {:>8.3} {:>10.3} {:>11.3} {:>6}",
            qu.name,
            n,
            yes as f64 / n.max(1) as f64,
            lookup_accuracy(&r1),
            lookup_accuracy(&r2),
            lookup_accuracy(&rb),
            lookup_accuracy(&rg),
            lookup_accuracy(&ra),
            "1.000"
        );
    }
    println!("\n  'counts' sees how many of each token so far (capped at 3), in no order. 'last+gap'");
    println!("  sees the last event and the order of magnitude of the silence before it; 'counts+age'");
    println!("  sees the counts and the order of magnitude of time since the case began. a question");
    println!("  these already answer is not one where a memory of when things happened can show.");
}
