# H1 tooling: interpretations of PROTOCOL_B5_1_HOLDOUT.md and review-driven changes

Recorded before any September 2026 data is downloaded. Manifest: code.commit = 89be2a4 (the merge of PR #19);
model SHA-256 are identical to the first freeze (5bdbc3e); B5.1 reproduced with 0 differences.

## Interpretations (where the protocol is ambiguous)
1. The manifest's code commit is the commit checked out at freeze time. Freeze was re-run at the merge commit 89be2a4.
2. The manifest lives on `exp/b5-1-holdout-run`, not in the tooling PR.
3. The FAA covered list is `busiest_airports(100)` over 2025-01..2026-08 (equal to the 100 METAR airports). `h1_airports.json` records METAR, weather and FAA lists.
4. The weather airport list is the 100 files in `data/weather/` (no app-only airports).
5. METAR origin coverage >= 85% is checked at both T-3h and T-1h.
6. "Parseable" FAA list page: longer than 500 characters and parses without error (zero GS/GDP rows is allowed). Weather check: every hour from the 1st 00:00 to the next month's 1st 23:00, no nulls in the 10 forecast columns. METAR "a report every day" is counted by UTC date of `valid`.
7. Outcome mapping: confirmed if the rule holds; else contradicted if the point estimate is >= 0 or ECE fails (stricter reading of the overlap); else not confirmed.
8. `score` builds its features fresh and writes no feature cache. The freeze cache is used for freeze only, in a fresh run folder each time.
9. The runtime airport index includes September airports; no feature encodes airport ids, so the frozen models are unaffected.
10. The B5.1-rate-window sensitivity also carries outcome labels, under `sensitivity_b5_1_rate_window`; it is non-gating and the report must say so.

## Review-driven changes (PR #19)
- An attempt record is written before any holdout data is loaded; a second `score` is refused without `--force-rerun`.
- `--force-rerun` needs `--reason` and writes `h1_b5_1_<month>_rerun<N>.json`; the original result is never overwritten.
- `score` refuses a dirty tree, code (feature/run files) that differs from the freeze commit by git hash, an airports file or models that differ from the manifest, and a manifest for another month.
- `check` writes a pass marker with the hashes of all holdout files; `score` requires it and re-verifies the hashes.
- Only 2026-09 can be unlocked (`holdout.UNLOCKABLE`, mirrored in `fetch_bts.sh`); later months stay refused until the section 7 policy is approved.
- `load_months` checks the unlock itself.
- Freeze always rebuilds from scratch (fresh cache folder).
- METAR and weather window-end checks are both `>=`.
