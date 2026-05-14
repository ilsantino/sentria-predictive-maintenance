# Roadmap

## Q2 2026 (current — completed)

- [x] Pydantic data contracts.
- [x] Synthetic telemetry generator (NASA C-MAPSS inspired).
- [x] Rolling-window feature extractor.
- [x] Isolation Forest detector (sklearn).
- [x] ARIMA baseline (statsmodels).
- [x] LSTM autoencoder (PyTorch).
- [x] Ensemble with three fusion strategies.
- [x] Event-level evaluation metrics with lead-time.
- [x] Walk-forward evaluation harness.
- [x] CLI (typer).
- [x] Pytest suite covering every module.
- [x] CI on GitHub Actions.
- [x] ARCHITECTURE / METHODOLOGY / EVALUATION / DATA_SCHEMA docs.

## Q3 2026 (pilot execution)

- [ ] **Week 1-2**: data agreement with client, instrumentation audit
  of the 3 pilot lines, definition of ground-truth labels with the
  client's QA team.
- [ ] **Week 3-4**: real ETL from line PLCs / sensors into S3 parquet,
  partitioned by `(machine_id, day)`. EDA notebook + data-quality
  report.
- [ ] **Week 5-6**: train all three baselines and the ensemble on the
  first two months of real data. Walk-forward validation on month 3.
- [ ] **Week 7-8**: ablations matrix (see `docs/EVALUATION.md`).
- [ ] **Week 9-10**: per-line threshold calibration on the validation
  slice. Simulate alert volumes against operator-availability budget.
- [ ] **Week 11-12**: shadow deployment (alerts logged but not
  delivered) on month 4 data to measure false-alarm rate without
  costing operator attention.
- [ ] **Week 13+**: go-live with one operator per line as the alert
  consumer. Collect operator feedback (Useful / Already known / False
  alarm / Confusing).

## Q4 2026 (model card + paper)

- [ ] Generate the model card following the template in
  `docs/EVALUATION.md`.
- [ ] Failure-mode analysis: cluster FPs and FNs, identify recurring
  patterns.
- [ ] Decide between **Production**, **Iteration**, or **Shelve**
  (criteria in `docs/EVALUATION.md`).
- [ ] Write technical report (~15-20 pages) covering motivation,
  related work, methodology, results, limitations, lessons.

## Q1 2027 (Phase 2 — if pilot succeeds)

- [ ] Expand to additional production lines.
- [ ] Add gradient-boosted residual learner on top of the ensemble.
- [ ] Integrate with SENTRIA core's incident escalation flow so an
  alert that goes unacknowledged for X minutes escalates to the next
  responsible person.
- [ ] Public technical writeup (workshop paper or company blog).
- [ ] Open-source release pending client agreement on what is shared.

## Open questions

1. **Sensor instrumentation gaps**: which lines have insufficient
   sensor coverage for predictive maintenance to add value? Could
   targeted instrumentation upgrades (vibration sensors, oil-quality
   sensors) change the answer?
2. **Label provenance**: how much of the historical incident log is
   reconciled to "real failure" vs "operator-reported issue"? The
   ratio sets a ceiling on what evaluation precision means.
3. **Operator workload**: how many alerts per shift can each operator
   process? This is the binding constraint on false-alarm rate, not
   model performance.
4. **Closed-loop control**: under what specific conditions, if any,
   would automated shutdown be acceptable? (Default answer: never
   for this pilot. Revisit in Phase 2.)
