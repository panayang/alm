//! What does the user want now? Dialogue state, asked at every turn.
//!
//! **Open-loop instrument.** The world's next event here is fixed in advance
//! and ignores what the model says, so this measures memory and prediction,
//! not response. It is kept as a record; it is not the test this design is
//! for (see the top of `lib.rs`).
//!
//! BPI 2012 failed its screen: its structural questions are answered by the
//! last two events or by counts, and its time questions reduce to keeping a
//! clock. A task-oriented dialogue fails neither way. MultiWOZ 2.2: 10437
//! dialogues between a tourist and a clerk in Cambridge, each turn annotated
//! with its dialogue acts (Hotel-Inform area=centre, Train-Request leaveat, ...)
//! and, after each user turn, with the full state the user has asked for so
//! far. The question "what area does the user want the hotel in, now?" has
//! its answer in something said turns ago, possibly by the clerk and only
//! accepted, possibly revised since. A window of the last three turns answers
//! 0.68 of these questions, the bag of everything said 0.58; the rule "the
//! last value the user gave for that slot" answers 0.91 (joint over all slots
//! of a turn: 0.54). That rule is itself a program over the history --
//! retrieval of the latest binding addressed by domain and slot, keeping who
//! said it -- which is what this architecture's content-addressed superposed
//! memory is for.
//!
//! The stream is the dialogue acts, not the words: speaker, act, slot, value.
//! Values of slots the user states are tokens; the clerk's reference numbers,
//! phone numbers, addresses and the like are one token, "information". After
//! each user turn one slot is asked about, and the annotated state answers as
//! an event of the world. The question is put in a side room: the dialogue's
//! own situation is saved before and restored after, so what is written by
//! the answer reaches long-term memory but the answer does not become part of
//! the conversation it was about -- otherwise "what was answered last time"
//! would stand in for the history.
//!
//! Memory is continual across dialogues and never reset; the situation is
//! cleared at the start of each dialogue. The dialogues run train, dev, test
//! in file order, and every answer is scored prequentially.

use crate::config::Config;
use crate::model::Model;
use std::collections::HashMap;

// ---- a small JSON reader: enough for these files, nothing more ------------

pub enum J {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        match self {
            J::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, b)| b),
            _ => None,
        }
    }
    pub fn str(&self) -> &str {
        match self {
            J::Str(s) => s,
            _ => "",
        }
    }
    pub fn arr(&self) -> &[J] {
        match self {
            J::Arr(v) => v,
            _ => &[],
        }
    }
    pub fn obj(&self) -> &[(String, J)] {
        match self {
            J::Obj(v) => v,
            _ => &[],
        }
    }
}

struct P<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> P<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && (self.b[self.i] as char).is_ascii_whitespace() {
            self.i += 1;
        }
    }
    fn value(&mut self) -> J {
        self.ws();
        match self.b[self.i] {
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b'}' {
                        self.i += 1;
                        return J::Obj(v);
                    }
                    let k = self.string();
                    self.ws();
                    self.i += 1; // ':'
                    let x = self.value();
                    v.push((k, x));
                    self.ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b']' {
                        self.i += 1;
                        return J::Arr(v);
                    }
                    v.push(self.value());
                    self.ws();
                    if self.b[self.i] == b',' {
                        self.i += 1;
                    }
                }
            }
            b'"' => J::Str(self.string()),
            b't' => {
                self.i += 4;
                J::Bool(true)
            }
            b'f' => {
                self.i += 5;
                J::Bool(false)
            }
            b'n' => {
                self.i += 4;
                J::Null
            }
            _ => {
                let s = self.i;
                while self.i < self.b.len() && b"+-0123456789.eE".contains(&self.b[self.i]) {
                    self.i += 1;
                }
                J::Num(std::str::from_utf8(&self.b[s..self.i]).unwrap().parse().unwrap_or(0.0))
            }
        }
    }
    fn string(&mut self) -> String {
        self.i += 1; // opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' | b'f' => {}
                        b'u' => {
                            let h = std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap();
                            self.i += 4;
                            let cp = u32::from_str_radix(h, 16).unwrap_or(0x3f);
                            let ch = char::from_u32(cp).unwrap_or('?');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        x => out.push(x),
                    }
                }
                x => out.push(x),
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

pub fn parse(text: &str) -> J {
    P { b: text.as_bytes(), i: 0 }.value()
}

fn read_json(path: &str) -> J {
    parse(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {}", path, e)))
}

// ---- the corpus ------------------------------------------------------------

/// Slots whose values the user states and the state records. Every other
/// slot's value -- a reference number, a phone number, a fee -- is one token.
const STATE_SLOTS: [&str; 18] = [
    "area", "pricerange", "type", "parking", "internet", "stars", "name", "food", "bookday",
    "bookpeople", "bookstay", "booktime", "day", "departure", "destination", "leaveat", "arriveby",
    "department",
];

pub struct Turn {
    pub user: bool,
    /// (act, [(slot, value or "")]) in annotation order.
    pub acts: Vec<(String, Vec<(String, String)>)>,
    /// After a user turn: "domain-slot" -> accepted values.
    pub state: Vec<(String, Vec<String>)>,
    /// The domain the user is acting in, where the annotation says so.
    pub active: Option<String>,
}

pub struct Dialogue {
    pub id: String,
    pub turns: Vec<Turn>,
}

fn norm(v: &str) -> String {
    v.trim().to_lowercase()
}

pub fn load(dir: &str, limit: usize) -> Vec<Dialogue> {
    eprintln!("  reading dialog acts ...");
    let acts = read_json(&format!("{}/dialog_acts.json", dir));
    let mut out = Vec::new();
    for split in ["train", "dev", "test"] {
        let mut files: Vec<String> = std::fs::read_dir(format!("{}/{}", dir, split))
            .unwrap_or_else(|e| panic!("{}/{}: {}", dir, split, e))
            .filter_map(|e| e.ok())
            .map(|e| e.path().to_string_lossy().into_owned())
            .filter(|p| p.ends_with(".json"))
            .collect();
        files.sort();
        for f in files {
            eprintln!("  reading {} ...", f);
            let j = read_json(&f);
            for d in j.arr() {
                let id = d.get("dialogue_id").map(|x| x.str().to_string()).unwrap_or_default();
                let services: Vec<&str> =
                    d.get("services").map(|s| s.arr().iter().map(|x| x.str()).collect()).unwrap_or_default();
                let da = acts.get(&id);
                let mut turns = Vec::new();
                for t in d.get("turns").map(|x| x.arr()).unwrap_or(&[]) {
                    let tid = t.get("turn_id").map(|x| x.str()).unwrap_or("");
                    let user = t.get("speaker").map(|x| x.str()) == Some("USER");
                    let mut ta = Vec::new();
                    if let Some(a) = da.and_then(|a| a.get(tid)).and_then(|a| a.get("dialog_act")) {
                        for (name, pairs) in a.obj() {
                            let mut sv = Vec::new();
                            for p in pairs.arr() {
                                let pa = p.arr();
                                if pa.len() < 2 {
                                    continue;
                                }
                                let v = pa[1].str();
                                let v = if v == "?" || v == "none" { String::new() } else { norm(v) };
                                sv.push((pa[0].str().to_string(), v));
                            }
                            ta.push((name.clone(), sv));
                        }
                    }
                    let mut state = Vec::new();
                    let mut active = None;
                    if user {
                        for fr in t.get("frames").map(|x| x.arr()).unwrap_or(&[]) {
                            let svc = fr.get("service").map(|x| x.str()).unwrap_or("");
                            let Some(st) = fr.get("state") else { continue };
                            if services.contains(&svc)
                                && st.get("active_intent").map(|x| x.str()).unwrap_or("NONE") != "NONE"
                            {
                                active = Some(svc.to_string());
                            }
                            if let Some(sv) = st.get("slot_values") {
                                for (k, vs) in sv.obj() {
                                    state.push((k.clone(), vs.arr().iter().map(|x| norm(x.str())).collect()));
                                }
                            }
                        }
                    }
                    turns.push(Turn { user, acts: ta, state, active });
                }
                out.push(Dialogue { id, turns });
                if limit > 0 && out.len() >= limit {
                    return out;
                }
            }
        }
    }
    out
}

// ---- tokens ------------------------------------------------------------------

#[derive(Default)]
struct Vocab {
    id: HashMap<String, usize>,
    names: Vec<String>,
}

impl Vocab {
    fn tok(&mut self, s: &str) -> usize {
        if let Some(&i) = self.id.get(s) {
            return i;
        }
        self.names.push(s.to_string());
        self.id.insert(s.to_string(), self.names.len() - 1);
        self.names.len() - 1
    }
}

/// An act as a token, marked with who performed it. The clerk informs with
/// the same acts the user does (Hotel-Inform: 3708 by users, 2367 by the
/// clerk, in the first three training files), so without the mark "what did
/// the user say about the area" and "what the clerk said about it" share a key.
fn said_by(user: bool, act: &str) -> String {
    format!("{}:{}", if user { "U" } else { "S" }, act)
}

/// The domain a mention belongs to: the act's own, or for a Booking act the
/// one the user is acting in. General acts carry no slot.
fn mention_key(act: &str, slot: &str, active: &Option<String>) -> Option<String> {
    if !STATE_SLOTS.contains(&slot) {
        return None;
    }
    let dom = act.split('-').next().unwrap_or("").to_lowercase();
    let dom = if dom == "booking" { active.clone()? } else { dom };
    if dom == "general" {
        return None;
    }
    Some(format!("{}-{}", dom, slot))
}

/// One asked question, and what everyone answered.
struct Asked {
    dialogue: usize,
    /// Turn pairs since the user last gave a value for this slot; None if never.
    gap: Option<usize>,
    set: bool,
    ours: bool,
    /// The same question with the dialogue wiped, nothing written: what the
    /// answer owes to long-term memory alone.
    wiped: Option<bool>,
    conf: f64,
    p_gold: f64,
    changes: u32,
    judges: [bool; 5],
    /// What a wrong answer was, if it was wrong: see WRONG.
    wrong: Option<usize>,
}

const JUDGES: [&str; 5] =
    ["none", "user, last 3 pairs", "bag (most said)", "last mention", "last user mention"];
const SILENCE: usize = 6;
/// A wrong answer was: none; a value this slot had earlier in the dialogue;
/// this slot's value in another domain; a value said here for another slot;
/// what memory alone answers; anything else.
const WRONG: [&str; 6] = ["none", "stale", "other domain", "other slot", "prior", "other"];

/// Build the stream, put the questions, and compare with the judges on
/// exactly the same questions.
pub fn run(dir: &str, limit: usize, d: usize, banks: usize, seed: u64, eta: Option<f32>, horizon: Option<f32>) {
    let dl = load(dir, limit);
    let mut v = Vocab::default();
    let (u_tok, s_tok, q_tok, info, none) =
        (v.tok("<user>"), v.tok("<clerk>"), v.tok("<ask>"), v.tok("<information>"), v.tok("<none>"));
    // Declared answer sets: every value the state ever gives a slot, and none.
    let mut answers: HashMap<String, Vec<usize>> = HashMap::new();
    for dg in &dl {
        for t in &dg.turns {
            for (k, vs) in &t.state {
                let slot = k.split('-').nth(1).unwrap_or("");
                let dom = k.split('-').next().unwrap_or("");
                v.tok(&format!("#{}", dom));
                v.tok(slot);
                for x in vs {
                    let tk = v.tok(&format!("={}", x));
                    let e = answers.entry(k.clone()).or_insert_with(|| vec![none]);
                    if !e.contains(&tk) {
                        e.push(tk);
                    }
                }
            }
            for (a, sv) in &t.acts {
                v.tok(&said_by(t.user, a));
                for (s, x) in sv {
                    v.tok(s);
                    if !x.is_empty() && STATE_SLOTS.contains(&s.as_str()) {
                        v.tok(&format!("={}", x));
                    }
                }
            }
        }
    }
    let vocab = v.names.len();
    eprintln!(
        "  {} dialogues, {} tokens in the vocabulary, {} slots asked about",
        dl.len(),
        vocab,
        answers.len()
    );

    let mut cfg = Config::local();
    cfg.seed = seed;
    cfg.vocab = vocab;
    cfg.d = d;
    cfg.cleanup_floor_mult = 1.1;
    cfg.mem_banks = banks;
    if let Some(e) = eta {
        cfg.eta = e;
    }
    if let Some(h) = horizon {
        cfg.horizon = h;
    }
    cfg.apply_env();
    cfg.derive();
    let mut m = Model::new(cfg);
    let blank = m.volatile();
    let mut asked: Vec<Asked> = Vec::new();
    let mut qn = 0u64;
    let t0 = std::time::Instant::now();
    let n = dl.len();
    for (di, dg) in dl.iter().enumerate() {
        if di > 0 && di % (n / 20).max(1) == 0 {
            let el = t0.elapsed().as_secs_f64();
            let r: Vec<&Asked> = asked.iter().rev().take(2000).collect();
            let acc = r.iter().filter(|a| a.ours).count() as f64 / r.len().max(1) as f64;
            let lum = r.iter().filter(|a| a.judges[4]).count() as f64 / r.len().max(1) as f64;
            eprintln!(
                "  {}/{} dialogues, {:.0}s, ~{:.0}s left; last 2000 asked: ours {:.3}, last user mention {:.3}",
                di,
                n,
                el,
                el * (n - di) as f64 / di as f64,
                acc,
                lum
            );
        }
        m.restore(blank.clone());
        // (turn, by the user, key, value token)
        let mut hist: Vec<(usize, bool, String, usize)> = Vec::new();
        let mut active: Option<String> = None;
        for (ti, t) in dg.turns.iter().enumerate() {
            if t.active.is_some() {
                active = t.active.clone();
            }
            m.tick(Some(if t.user { u_tok } else { s_tok }), false);
            for (a, sv) in &t.acts {
                let at = v.id[&said_by(t.user, a)];
                if sv.is_empty() {
                    m.tick(Some(at), false);
                }
                // The act before every slot, not once per act: a value is
                // then always preceded by the same (act, slot) pair, whichever
                // position it had in the act. Written once, 41% of the values
                // a user gives followed (previous value, slot) instead.
                for (s, x) in sv {
                    m.tick(Some(at), false);
                    m.tick(Some(v.id[s]), false);
                    if x.is_empty() {
                        continue;
                    }
                    let vt = if STATE_SLOTS.contains(&s.as_str()) { v.id[&format!("={}", x)] } else { info };
                    m.tick(Some(vt), false);
                    if let Some(k) = mention_key(a, s, &active) {
                        hist.push((ti, t.user, k, vt));
                    }
                }
            }
            if !t.user {
                continue;
            }
            // Which slot to ask about: anything set, or anything mentioned.
            let mut keys: Vec<String> = t.state.iter().map(|(k, _)| k.clone()).collect();
            for h in &hist {
                if !keys.contains(&h.2) && answers.contains_key(&h.2) {
                    keys.push(h.2.clone());
                }
            }
            keys.retain(|k| answers.contains_key(k));
            if keys.is_empty() {
                continue;
            }
            keys.sort();
            let k = keys[crate::num::uniform_below(seed ^ 0x3057, qn, keys.len() as u64) as usize].clone();
            qn += 1;
            let gold: Vec<usize> = t
                .state
                .iter()
                .find(|(a, _)| *a == k)
                .map(|(_, vs)| vs.iter().map(|x| v.id[&format!("={}", x)]).collect())
                .unwrap_or_else(|| vec![none]);
            let cand = &answers[&k];

            // The judges, on the same question.
            let pick = |f: &dyn Fn(&(usize, bool, String, usize)) -> bool| {
                hist.iter().rev().find(|h| h.2 == k && f(h)).map(|h| h.3).unwrap_or(none)
            };
            let bag = {
                let mut c: HashMap<usize, usize> = HashMap::new();
                let mut best = (0usize, none);
                for h in hist.iter().filter(|h| h.2 == k) {
                    let e = c.entry(h.3).or_insert(0);
                    *e += 1;
                    if *e > best.0 {
                        best = (*e, h.3);
                    }
                }
                best.1
            };
            let ja = [
                none,
                pick(&|h| h.1 && h.0 + 5 >= ti),
                bag,
                pick(&|_| true),
                pick(&|h| h.1),
            ];
            let gap = hist.iter().rev().find(|h| h.2 == k && h.1).map(|h| (ti - h.0) / 2);

            // Asked in a side room.
            let saved = m.volatile();
            let dom = k.split('-').next().unwrap();
            let slot = k.split('-').nth(1).unwrap();
            // The question ends on the pair a user sets this slot with, when
            // that act exists -- "Hotel-Inform area" -- so that what follows it
            // in this dialogue is what is being asked for. Otherwise the domain.
            let mut cap = dom.to_string();
            if let Some(c) = cap.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            let inform = said_by(true, &format!("{}-Inform", cap));
            let dom_tok = v.id.get(&inform).copied().unwrap_or(v.id[&format!("#{}", dom)]);
            let q = [q_tok, dom_tok, v.id[slot]];
            let read = |m: &Model| {
                let sc = m.spread_now();
                let ps: Vec<f64> = cand.iter().map(|&c| sc.prob_of(&m.store, c as u32) as f64).collect();
                let z: f64 = ps.iter().sum::<f64>().max(1e-30);
                let (bi, bp) = ps.iter().enumerate().fold((0, -1.0), |b, (i, &p)| if p > b.1 { (i, p) } else { b });
                let pg: f64 = cand.iter().zip(&ps).filter(|(c, _)| gold.contains(c)).map(|(_, p)| p).sum();
                (cand[bi], bp / z, pg / z)
            };
            // First with nothing of this dialogue, writing nothing -- on every
            // fourth question only, which is enough to read it and halves the
            // cost. The others are marked unasked.
            let ask_wiped = qn % 4 == 1;
            let mut wiped_top = usize::MAX;
            if ask_wiped {
                m.restore(blank.clone());
                m.frozen = true;
                for &x in &q {
                    m.tick(Some(x), false);
                }
                for _ in 0..SILENCE {
                    m.tick(None, false);
                }
                wiped_top = read(&m).0;
                m.frozen = false;
            }
            let wiped = if ask_wiped { Some(gold.contains(&wiped_top)) } else { None };
            m.restore(saved.clone());
            for &x in &q {
                m.tick(Some(x), false);
            }
            let mut prev = read(&m).0;
            let mut changes = 0;
            for _ in 0..SILENCE {
                m.tick(None, false);
                let now = read(&m).0;
                if now != prev {
                    changes += 1;
                    prev = now;
                }
            }
            let (top, conf, pg) = read(&m);
            m.tick(Some(gold[0]), false);
            m.restore(saved);

            asked.push(Asked {
                dialogue: di,
                gap,
                set: gold[0] != none,
                ours: gold.contains(&top),
                wiped,
                conf,
                p_gold: pg,
                changes,
                judges: std::array::from_fn(|j| gold.contains(&ja[j])),
                wrong: if gold.contains(&top) {
                    None
                } else if top == none {
                    Some(0)
                } else if hist.iter().any(|h| h.2 == k && h.3 == top) {
                    Some(1)
                } else if hist.iter().any(|h| h.3 == top && h.2.split('-').nth(1) == k.split('-').nth(1)) {
                    Some(2)
                } else if hist.iter().any(|h| h.3 == top) {
                    Some(3)
                } else if top == wiped_top {
                    Some(4)
                } else {
                    Some(5)
                },
            });
        }
    }
    report(&asked, n);
}

fn report(asked: &[Asked], n: usize) {
    let row = |name: &str, sel: &dyn Fn(&Asked) -> bool| {
        let r: Vec<&Asked> = asked.iter().filter(|a| sel(a)).collect();
        if r.is_empty() {
            return;
        }
        let f = |g: &dyn Fn(&Asked) -> bool| r.iter().filter(|a| g(a)).count() as f64 / r.len() as f64;
        let conf = r.iter().map(|a| a.conf).sum::<f64>() / r.len() as f64;
        let brier = r.iter().map(|a| (1.0 - a.p_gold).powi(2)).sum::<f64>() / r.len() as f64;
        let wr: Vec<bool> = r.iter().filter_map(|a| a.wiped).collect();
        let wf = wr.iter().filter(|&&x| x).count() as f64 / wr.len().max(1) as f64;
        print!("{:>24} {:>7} {:>6.3} {:>6.3} {:>6.3} {:>6.3}", name, r.len(), f(&|a| a.ours), wf, conf, brier);
        for j in 0..JUDGES.len() {
            print!(" {:>8.3}", f(&|a| a.judges[j]));
        }
        println!();
    };
    println!("\nwhat does the user want now: accuracy on the asked slot, prequential\n");
    print!("{:>24} {:>7} {:>6} {:>6} {:>6} {:>6}", "questions", "n", "ours", "wiped", "conf", "brier");
    for j in JUDGES {
        print!(" {:>8}", &j[..j.len().min(8)]);
    }
    println!("\n  (judges: {})", JUDGES.join(" | "));
    row("all", &|_| true);
    for q in 0..4 {
        let (a, b) = (n * q / 4, n * (q + 1) / 4);
        row(&format!("dialogues {}-{}", a, b), &|x| x.dialogue >= a && x.dialogue < b);
    }
    row("answer is none", &|a| !a.set);
    row("answer is set", &|a| a.set);
    row("  user never said it", &|a| a.set && a.gap.is_none());
    for g in 0..7 {
        row(
            &format!("  said {} pairs ago{}", g, if g == 6 { "+" } else { "" }),
            &|a| a.set && a.gap.map(|x| x.min(6)) == Some(g),
        );
    }
    row("answer changed in silence", &|a| a.changes > 0);
    println!("
  what a wrong answer was, among wrong answers ('prior' is known only where the wiped arm was asked, one question in four)");
    for (name, sel) in [
        ("all", &(|_: &Asked| true) as &dyn Fn(&Asked) -> bool),
        ("user said it this turn", &|a: &Asked| a.set && a.gap == Some(0)),
        ("answer is none", &|a: &Asked| !a.set),
    ] {
        let w: Vec<usize> = asked.iter().filter(|a| sel(a)).filter_map(|a| a.wrong).collect();
        print!("  {:>24} {:>6}", name, w.len());
        for (i, lab) in WRONG.iter().enumerate() {
            print!("  {} {:.3}", lab, w.iter().filter(|&&x| x == i).count() as f64 / w.len().max(1) as f64);
        }
        println!();
    }
    println!("\n  'ours' is the value most probable among the slot's declared answers after the");
    println!("  question and {} silent ticks; conf is that probability, brier is on the gold", SILENCE);
    println!("  answer's probability. 'user, last 3 pairs' is the last-user-mention rule with");
    println!("  a window. the judges resolve a Booking act's domain from the annotated active");
    println!("  intent, which the model does not see.");
}
