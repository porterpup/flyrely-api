# Protocol addendum, November 2026

Amends `PROTOCOL.md` without rewriting it. Recorded 2026-11-03.

1. **Data source.** The data moved to the all-carrier BTS Marketing Carrier file (`parquet_v2`), about 9.5% more flights than the Reporting Carrier file, with the operating carrier mapped to the reporting columns.
2. **Committed-model features.** The committed model's features now use the operating carrier.
3. **2026-01 to 2026-08 is validation.** That period has been viewed many times (the b4 diagnostic and others), so it no longer counts as untouched. Treat it as validation data.
4. **Clean holdout.** The only untouched holdout is 2026-09 onward. It stays protected by the lock: `fetch_bts.sh` and `fetch_weather.py` refuse it unless `FLYRELY_HOLDOUT_UNLOCK=1`.
5. **Registered result.** `results/results.json` is the registered result and must not be regenerated over. `backtest.py` now writes `results/results_v2data.json` (with matching `calibration_v2data.png` and `booking_model_candidate_v2data.joblib`) and leaves the original files intact.
