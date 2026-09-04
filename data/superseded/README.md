# Raw run outputs — superseded

**These numbers are void.** They are kept because the project's record includes
what was measured before each defect was found, and because several of them are
the evidence for the method lessons in `REPORT.md` §6 and `paper.tex` §9 — a run
in which six arms came out bit-identical is itself a finding.

Each of the following was invalidated by a change to the mechanism or to the
generator, and re-run rather than carried forward:

- runs predating the **competitive-negatives** fix measured a readout that had
  never been taught to separate the candidates it had to choose between, so
  every conjunction number in them is carried by re-lookup;
- runs predating the **unitary codebook** measured a binding whose inverse lost
  almost half the signal with a single item stored;
- runs predating **identity-hashed banks** measured an address that made a near
  miss undetectable by construction;
- runs predating the **source repairs** measured a Latin square that was not
  zero-marginal (a single-cue predictor reached 0.2745), composition queries
  that shared tokens with first-order facts, a fixed silence length, and a
  final-exam burst whose background never appeared elsewhere in the stream;
- runs predating the **`bind_decay` fix** measured a binding trace that retained
  1.6% of itself by the time anything read it.

Do not cite anything here as a result. `data/current/` holds the runs the paper
and report rest on.
