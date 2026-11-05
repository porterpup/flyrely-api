# B5 results: observed METAR and FAA GS/GDP programs in the run-up models

**Status:** `backtest/experiments_b5.py` ran as pre-registered in `PROTOCOL_B5.md`, including both amendments (both
committed before any evaluation). No further amendments were made. Raw numbers are in `backtest/results/b5_metar_faa.json`.

**Validation only.** The evaluation months are 2026-01..08, which are validation months
(`PROTOCOL_ADDENDUM_2026-11.md`), not a clean holdout. Nothing dated 2026-09 or later was loaded, and
`FLYRELY_HOLDOUT_UNLOCK` was never set. 2026-09 onward stays the clean holdout for confirmation.

Data: 5,227,724 flights in the evaluation period, of which 5,108,811 were flown.

## Decision rule (restated from the protocol)

An arm is adopted at a horizon only if both of these hold:
- its Brier difference from the base is negative, with the 95% day-block bootstrap CI entirely below zero;
- it improves Brier in most evaluation months.

The combined A+B arm is reported for information only and does not gate adoption.

## Verdict

| Source | T-1d | T-3h | T-1h |
|---|---|---|---|
| A. METAR (observed weather) | not tested (per protocol) | **adopt** | **adopt** |
| B. FAA GS/GDP programs | **adopt** | **adopt** | **adopt** |

All five gating tests pass both rules.

## Results by horizon

Brier difference = Brier(arm) minus Brier(base). Negative is better.

| Horizon | Arm | AUC | Brier | ECE | Brier diff [95% CI] | Months improved | Adopt |
|---|---|---|---|---|---|---|---|
| T-1d | base | 0.6985 | 0.16437 | 0.0182 | | | |
| T-1d | +B | 0.7013 | 0.16387 | 0.0194 | -0.000504 [-0.000679, -0.000334] | 7/8 | yes |
| T-3h | base | 0.8220 | 0.12299 | 0.0057 | | | |
| T-3h | +A | 0.8242 | 0.12243 | 0.0062 | -0.000564 [-0.000649, -0.000478] | 8/8 | yes |
| T-3h | +B | 0.8274 | 0.12087 | 0.0037 | -0.002127 [-0.002324, -0.001922] | 8/8 | yes |
| T-3h | +A+B (not gating) | 0.8283 | 0.12071 | 0.0044 | -0.002280 [-0.002474, -0.002080] | 8/8 | n/a |
| T-1h | base | 0.8552 | 0.10667 | 0.0080 | | | |
| T-1h | +A | 0.8572 | 0.10613 | 0.0078 | -0.000531 [-0.000620, -0.000444] | 8/8 | yes |
| T-1h | +B | 0.8620 | 0.10408 | 0.0062 | -0.002581 [-0.002821, -0.002337] | 8/8 | yes |
| T-1h | +A+B (not gating) | 0.8632 | 0.10382 | 0.0066 | -0.002848 [-0.003102, -0.002590] | 8/8 | n/a |

For context, the committed booking model (`bts18-gb-calib-v2`) on the same rows scores AUC 0.6774, Brier 0.17005,
ECE 0.0453.

## Effect size in plain terms

- **METAR** gives a small but very consistent gain. Brier falls by about 0.0005 (roughly 0.4% relative) and AUC
  rises by about 0.002.
- **FAA programs** give the larger gain. Brier falls by 0.0021 at T-3h and 0.0026 at T-1h (about 2% relative), and
  AUC rises by 0.005 to 0.007. At T-1d the gain is 0.0005.
- **Combining them** adds only 0.0001 to 0.0003 over FAA alone.
- **The FAA gain grows through the year.** At T-1h it goes from -0.0014 in January to -0.0044 in August.

## Brier difference by month

| Month | T-1d +B | T-3h +A | T-3h +B | T-3h +A+B | T-1h +A | T-1h +B | T-1h +A+B |
|---|---|---|---|---|---|---|---|
| 2026-01 | -0.00048 | -0.00068 | -0.00117 | -0.00167 | -0.00054 | -0.00140 | -0.00164 |
| 2026-02 | +0.00015 | -0.00071 | -0.00094 | -0.00146 | -0.00026 | -0.00104 | -0.00131 |
| 2026-03 | -0.00051 | -0.00052 | -0.00163 | -0.00180 | -0.00035 | -0.00197 | -0.00218 |
| 2026-04 | -0.00048 | -0.00053 | -0.00195 | -0.00197 | -0.00036 | -0.00231 | -0.00252 |
| 2026-05 | -0.00087 | -0.00037 | -0.00213 | -0.00204 | -0.00048 | -0.00237 | -0.00260 |
| 2026-06 | -0.00059 | -0.00052 | -0.00256 | -0.00276 | -0.00066 | -0.00314 | -0.00345 |
| 2026-07 | -0.00076 | -0.00062 | -0.00290 | -0.00295 | -0.00078 | -0.00363 | -0.00400 |
| 2026-08 | -0.00037 | -0.00060 | -0.00340 | -0.00337 | -0.00077 | -0.00436 | -0.00466 |

## Coverage and FAA prevalence

**Coverage on evaluation flights:**
- METAR covers 93 of the 100 airports. That is 86.9% of flights at the origin and 75.7% at both ends.
- FAA covers all 100 airports. That is 89.5% of flights at the origin and 79.8% at both ends.

**FAA prevalence at the T-3h cutoff:**
- At the origin, a ground stop affects 1.3% of flights and a GDP in force 2.9%.
- At the destination, the figures are 1.0% and 2.3%.

At T-1d the `gdp_prop` and `gdp_cover` flags are always zero because of the default periods. Only `gs_now` and
`gdp_now` carry signal there.

## Caveats

1. **Validation, not holdout.** Results are on validation months 2026-01..08 and should be confirmed on 2026-09 onward.
2. **FAA program periods are defaults, not real periods.** Amendment 2 set them as GS 90 min and GDP 480 min, based
   on a sample of 230 detail pages. 24,331 of 27,045 advisories used the default. Live serving would know the real
   periods, so behaviour in production could differ.
3. **FAA probably partly acts as a "system already disrupted" signal.** It uses only information stamped at or before
   each cutoff. In production it needs a reliable live FAA feed; if the feed is missing, predictions fall back to base.
4. **METAR gaps.** 7 of the 100 airports have no METAR station, and flights there get missing values. Coverage at both
   ends is 76 to 80%.
5. **METAR horizons.** METAR was tested only at T-3h and T-1h, as the protocol specified.
6. **METAR's gain is tiny.** Whether it justifies a live aviationweather.gov dependency is a product decision. FAA is
   the more valuable source.
7. **Multiple comparisons.** With five gating tests, the CI bound nearest zero is T-1d FAA at -0.00033, so correcting
   for multiple comparisons does not change the verdict.
8. **Rebuilt cache.** A truncated feature cache left over from 2026-11-03 crashed the first run. It was moved aside to
   `data/cache/b5_stale_20261103_truncated` (not deleted), and all features were rebuilt from scratch.

## Reproduce

```
venv/bin/python backtest/experiments_b5.py
```

The full run takes about 1 h 15 min.
