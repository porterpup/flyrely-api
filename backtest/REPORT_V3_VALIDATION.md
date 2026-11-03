# B1 validation: trailing recent-performance features and rolling recalibration

Status: results below were produced by `backtest/experiments_v3.py` (rule committed in ea49a5f before the run).

Validation only: 2026-01..08 is validation per `PROTOCOL_ADDENDUM_2026-11.md`. Months >= 2026-09 are never loaded
(the script hard-fails on them).

## Selection rule (fixed before running)

1. Candidates are servable variants only: data gap G >= 90 days (G=90, 120), each with recalibration K in {none, 1, 2, 3}.
   G=1/31/60 are diagnostics and cannot be chosen.
2. Chosen variant = lowest walk-forward Brier on flown flights, 2026-01..08.
3. Production adoption also requires the Brier difference (chosen minus committed model, scored as in `b4_diagnostic`
   with haversine distance) on 2026-01..08 to have a 95% day-block bootstrap interval (1,000 resamples, seed 42)
   entirely below 0. Otherwise: do not adopt.

Gap convention: the trailing window for departure date D covers days (D-G-W, D-G]; G=1 means yesterday is the newest
day used. The leakage test asserts no feature uses a day later than D-G.

## Results

Reproduce: `venv/bin/python backtest/experiments_v3.py --gaps 1,31,60,90,120 --full-gaps 1,90,120 --sample 0.2` (these are now the defaults; the committed run used exactly this).

**Verdict: DO NOT ADOPT.** Chosen servable variant V90|K=3; Brier minus committed +0.00106, 95% CI [+0.00033, +0.00174]. Trailing features help only with a live feed (G=1 AUC 0.696 vs 0.677 for B1) and the lagged (G>=90) feed adds nothing over the baseline; rolling recalibration fixes ECE (0.04 to about 0.015) but not Brier relative to the committed model.

Runtime: about 27 min wall for the first pass (lean-table + features + 7 model fits, 06:48-07:11) plus 3.2 min for the final cached evaluation pass; well inside the 2 h box. Sample fraction for non-full gaps: 0.2; full-data gaps: [1, 90, 120]. Fit: weights 2025-01..09, isotonic 2025-10..12, rates from prior 12 months.

### Grid: AUC / Brier / ECE, overall 2026-01..08 (flown flights)

| Variant | K | AUC | Brier | ECE | worst-month ECE | Jan ECE | Jan Brier | Jan AUC |
|---|---|---|---|---|---|---|---|---|
| committed |  | 0.6765 | 0.1677 | 0.0053 | 0.0426 | 0.0155 | 0.1586 | 0.6192 |
| B0 | None | 0.6785 | 0.1695 | 0.0426 | 0.0651 | 0.0392 | 0.1605 | 0.6104 |
| B0 | 1 | 0.6746 | 0.1685 | 0.0121 | 0.0507 | 0.0286 | 0.1596 | 0.6104 |
| B0 | 2 | 0.6756 | 0.1684 | 0.0148 | 0.0615 | 0.0292 | 0.1597 | 0.6104 |
| B0 | 3 | 0.6762 | 0.1684 | 0.0165 | 0.0521 | 0.0308 | 0.1599 | 0.6104 |
| B1 | None | 0.6768 | 0.1698 | 0.0423 | 0.0650 | 0.0377 | 0.1605 | 0.6102 |
| B1 | 1 | 0.6730 | 0.1688 | 0.0118 | 0.0508 | 0.0257 | 0.1594 | 0.6102 |
| B1 | 2 | 0.6738 | 0.1687 | 0.0138 | 0.0621 | 0.0290 | 0.1596 | 0.6102 |
| B1 | 3 | 0.6743 | 0.1688 | 0.0154 | 0.0533 | 0.0319 | 0.1599 | 0.6102 |
| V1 | None | 0.6961 | 0.1667 | 0.0420 | 0.0837 | 0.0504 | 0.1592 | 0.6436 |
| V1 | 1 | 0.6863 | 0.1667 | 0.0071 | 0.0825 | 0.0416 | 0.1585 | 0.6436 |
| V1 | 2 | 0.6867 | 0.1667 | 0.0105 | 0.0697 | 0.0418 | 0.1588 | 0.6436 |
| V1 | 3 | 0.6880 | 0.1667 | 0.0142 | 0.0604 | 0.0424 | 0.1589 | 0.6436 |
| V31 * | None | 0.6863 | 0.1652 | 0.0803 | 0.1104 | 0.1079 | 0.1503 | 0.6393 |
| V31 * | 1 | 0.6672 | 0.1617 | 0.0183 | 0.1131 | 0.0678 | 0.1433 | 0.6393 |
| V31 * | 2 | 0.6683 | 0.1618 | 0.0183 | 0.0950 | 0.0656 | 0.1444 | 0.6393 |
| V31 * | 3 | 0.6695 | 0.1618 | 0.0216 | 0.0857 | 0.0595 | 0.1436 | 0.6393 |
| V60 * | None | 0.6769 | 0.1635 | 0.0536 | 0.0894 | 0.0613 | 0.1423 | 0.6293 |
| V60 * | 1 | 0.6672 | 0.1616 | 0.0116 | 0.0785 | 0.0397 | 0.1408 | 0.6293 |
| V60 * | 2 | 0.6660 | 0.1622 | 0.0170 | 0.0688 | 0.0414 | 0.1418 | 0.6293 |
| V60 * | 3 | 0.6666 | 0.1625 | 0.0183 | 0.0647 | 0.0360 | 0.1411 | 0.6293 |
| V90 | None | 0.6767 | 0.1702 | 0.0460 | 0.0798 | 0.0579 | 0.1631 | 0.6200 |
| V90 | 1 | 0.6700 | 0.1695 | 0.0141 | 0.0645 | 0.0645 | 0.1649 | 0.6200 |
| V90 | 2 | 0.6724 | 0.1691 | 0.0152 | 0.0623 | 0.0553 | 0.1634 | 0.6200 |
| V90 | 3 | 0.6751 | 0.1688 | 0.0163 | 0.0561 | 0.0507 | 0.1626 | 0.6200 |
| V120 | None | 0.6781 | 0.1700 | 0.0452 | 0.0784 | 0.0331 | 0.1598 | 0.6231 |
| V120 | 1 | 0.6718 | 0.1691 | 0.0127 | 0.0521 | 0.0358 | 0.1604 | 0.6231 |
| V120 | 2 | 0.6727 | 0.1690 | 0.0133 | 0.0737 | 0.0320 | 0.1599 | 0.6231 |
| V120 | 3 | 0.6741 | 0.1689 | 0.0166 | 0.0632 | 0.0303 | 0.1595 | 0.6231 |

B0 = current booking set (row's own Distance/CRSElapsedTime); B1 = serving-parity baseline; V<G> = B1 + trailing features at gap G; `*` = fit and scored on a 20% random day-sample only (not comparable row-for-row with the others; G=31 and 60 are diagnostics). K=None means no recalibration.

### Per-month Brier / ECE / AUC (no recalibration, and K=3)

- **B0|K=None**: 01: 0.1605/0.0392/0.6104; 02: 0.1558/0.0525/0.6344; 03: 0.1763/0.0188/0.6604; 04: 0.1545/0.0596/0.6620; 05: 0.1638/0.0651/0.6795; 06: 0.1781/0.0579/0.7015; 07: 0.1861/0.0398/0.7051; 08: 0.1768/0.0388/0.7026
- **B0|K=3**: 01: 0.1599/0.0308/0.6104; 02: 0.1541/0.0232/0.6344; 03: 0.1763/0.0188/0.6604; 04: 0.1536/0.0521/0.6620; 05: 0.1617/0.0427/0.6795; 06: 0.1758/0.0317/0.7015; 07: 0.1856/0.0283/0.7051; 08: 0.1767/0.0271/0.7026
- **B1|K=None**: 01: 0.1605/0.0377/0.6102; 02: 0.1554/0.0496/0.6353; 03: 0.1766/0.0191/0.6586; 04: 0.1553/0.0604/0.6569; 05: 0.1640/0.0650/0.6778; 06: 0.1786/0.0572/0.6987; 07: 0.1865/0.0371/0.7028; 08: 0.1773/0.0374/0.7001
- **B1|K=3**: 01: 0.1599/0.0319/0.6102; 02: 0.1537/0.0208/0.6353; 03: 0.1766/0.0191/0.6586; 04: 0.1543/0.0533/0.6569; 05: 0.1620/0.0439/0.6778; 06: 0.1762/0.0296/0.6987; 07: 0.1861/0.0294/0.7028; 08: 0.1772/0.0267/0.7001
- **V1|K=None**: 01: 0.1592/0.0504/0.6436; 02: 0.1588/0.0837/0.6554; 03: 0.1743/0.0308/0.6741; 04: 0.1556/0.0744/0.6720; 05: 0.1584/0.0523/0.7036; 06: 0.1714/0.0373/0.7218; 07: 0.1808/0.0275/0.7235; 08: 0.1726/0.0297/0.7196
- **V1|K=3**: 01: 0.1589/0.0424/0.6436; 02: 0.1556/0.0604/0.6554; 03: 0.1743/0.0308/0.6741; 04: 0.1529/0.0575/0.6720; 05: 0.1570/0.0374/0.7036; 06: 0.1728/0.0328/0.7218; 07: 0.1845/0.0538/0.7235; 08: 0.1741/0.0390/0.7196
- **V31|K=None**: 01: 0.1503/0.1079/0.6393; 02: 0.1533/0.1104/0.6586; 03: 0.1653/0.0519/0.6624; 04: 0.1602/0.1077/0.6805; 05: 0.1624/0.0851/0.6925; 06: 0.1866/0.0302/0.7196; 07: 0.1872/0.0422/0.7041; 08: 0.1729/0.0699/0.6920
- **V31|K=3**: 01: 0.1436/0.0595/0.6393; 02: 0.1487/0.0806/0.6586; 03: 0.1653/0.0519/0.6624; 04: 0.1517/0.0585/0.6805; 05: 0.1563/0.0334/0.6925; 06: 0.1960/0.0857/0.7196; 07: 0.1905/0.0608/0.7041; 08: 0.1684/0.0196/0.6920
- **V60|K=None**: 01: 0.1423/0.0613/0.6293; 02: 0.1451/0.0646/0.6552; 03: 0.1657/0.0353/0.6555; 04: 0.1627/0.0894/0.6523; 05: 0.1620/0.0688/0.6784; 06: 0.1933/0.0232/0.6920; 07: 0.1851/0.0283/0.7087; 08: 0.1697/0.0324/0.6876
- **V60|K=3**: 01: 0.1411/0.0360/0.6293; 02: 0.1440/0.0469/0.6552; 03: 0.1657/0.0353/0.6555; 04: 0.1573/0.0641/0.6523; 05: 0.1570/0.0184/0.6784; 06: 0.1962/0.0542/0.6920; 07: 0.1909/0.0647/0.7087; 08: 0.1712/0.0402/0.6876
- **V90|K=None**: 01: 0.1631/0.0579/0.6200; 02: 0.1539/0.0311/0.6346; 03: 0.1753/0.0104/0.6623; 04: 0.1567/0.0798/0.6665; 05: 0.1670/0.0794/0.6737; 06: 0.1786/0.0507/0.6943; 07: 0.1871/0.0423/0.7035; 08: 0.1762/0.0369/0.7065
- **V90|K=3**: 01: 0.1626/0.0507/0.6200; 02: 0.1532/0.0139/0.6346; 03: 0.1753/0.0104/0.6623; 04: 0.1534/0.0561/0.6665; 05: 0.1630/0.0498/0.6737; 06: 0.1768/0.0282/0.6943; 07: 0.1856/0.0237/0.7035; 08: 0.1769/0.0324/0.7065
- **V120|K=None**: 01: 0.1598/0.0331/0.6231; 02: 0.1526/0.0421/0.6535; 03: 0.1767/0.0163/0.6550; 04: 0.1591/0.0770/0.6458; 05: 0.1668/0.0784/0.6724; 06: 0.1787/0.0513/0.6942; 07: 0.1851/0.0419/0.7100; 08: 0.1764/0.0411/0.7058
- **V120|K=3**: 01: 0.1595/0.0303/0.6231; 02: 0.1513/0.0205/0.6535; 03: 0.1767/0.0163/0.6550; 04: 0.1566/0.0632/0.6458; 05: 0.1637/0.0577/0.6724; 06: 0.1765/0.0218/0.6942; 07: 0.1852/0.0293/0.7100; 08: 0.1773/0.0340/0.7058

### Selection (pre-registered)

Candidates (Brier): {'V90|K=None': 0.17024, 'V90|K=1': 0.16949, 'V90|K=2': 0.16908, 'V90|K=3': 0.16881, 'V120|K=None': 0.16995, 'V120|K=1': 0.16915, 'V120|K=2': 0.16903, 'V120|K=3': 0.16892}

Chosen: **V90|K=3**

Brier(chosen) - Brier(committed): 0.00106, 95% day-block bootstrap CI [0.00033, 0.00174] (243 days, 5,108,811 flights). **Adopt: NO** (needs CI entirely below 0).

- Chosen minus B1 (same K): 0.00006 CI [-0.00062, 0.00076]
- B1 (no recal) minus committed: 0.00202 CI [0.00101, 0.00302]

### Cost of the data gap: G=90 vs G=1 (value of a live recent-actuals feed)

| K | Brier G=1 | Brier G=90 | Brier(90) - Brier(1) [95% CI] | AUC G=1 | AUC G=90 | ECE G=1 | ECE G=90 |
|---|---|---|---|---|---|---|---|
| None | 0.1667 | 0.1702 | 0.00353 [0.00269, 0.00443] | 0.6961 | 0.6767 | 0.0420 | 0.0460 |
| 1 | 0.1667 | 0.1695 | 0.00274 [0.00174, 0.00374] | 0.6863 | 0.6700 | 0.0071 | 0.0141 |
| 2 | 0.1667 | 0.1691 | 0.00238 [0.00163, 0.00318] | 0.6867 | 0.6724 | 0.0105 | 0.0152 |
| 3 | 0.1667 | 0.1688 | 0.00214 [0.00142, 0.00289] | 0.6880 | 0.6751 | 0.0142 | 0.0163 |

### Serving without a flight number (flight-number features NaN at validation, same model, no recalibration)

- V1: AUC 0.6894, Brier 0.1680, ECE 0.0427
- V31: AUC 0.6854, Brier 0.1651, ECE 0.0794
- V60: AUC 0.6767, Brier 0.1635, ECE 0.0535
- V90: AUC 0.6761, Brier 0.1703, ECE 0.0455
- V120: AUC 0.6781, Brier 0.1699, ECE 0.0447

### Per-carrier ECE, six regional carriers (n, ECE, mean predicted vs observed)

| Carrier | chosen | B1 (no recal) | committed |
|---|---|---|---|
| 9E | 0.0380 (0.268 vs 0.234, n=164,010) | 0.0570 (0.291 vs 0.234, n=164,010) | 0.0167 (0.237 vs 0.234, n=164,010) |
| PT | 0.0469 (0.229 vs 0.185, n=99,821) | 0.0731 (0.258 vs 0.185, n=99,821) | 0.0287 (0.213 vs 0.185, n=99,821) |
| QX | 0.0303 (0.210 vs 0.189, n=62,281) | 0.0476 (0.233 vs 0.189, n=62,281) | 0.0077 (0.190 vs 0.189, n=62,281) |
| G7 | 0.0560 (0.304 vs 0.248, n=56,499) | 0.0594 (0.304 vs 0.248, n=56,499) | 0.0528 (0.276 vs 0.248, n=56,499) |
| C5 | 0.0630 (0.228 vs 0.166, n=54,314) | 0.1047 (0.270 vs 0.166, n=54,314) | 0.0397 (0.199 vs 0.166, n=54,314) |
| YV | 0.0753 (0.215 vs 0.143, n=46,827) | 0.1102 (0.253 vs 0.143, n=46,827) | 0.0601 (0.203 vs 0.143, n=46,827) |

### Leakage test and caveats

- `tests/test_trailing_features.py` rebuilds the counters from only days <= D-G and flips every outcome after D-G on
  synthetic data; features are unchanged (G in 1, 31, 90, 120). The brief said "no day >= departure - G"; with the G=1
  = yesterday convention the newest permitted day is D-G itself, so the test asserts no day > D-G.
- HGB weights are fit on 2025-01..09 and isotonic on 2025-10..12 (backtest.py's fit/val split inside 2025), rather than
  a single 2025-01..12 fit. Recalibration for early 2026 months uses 2025 predictions that are in-sample: with K>=1, January 2026 recalibrates on 2025-10, which lies inside the isotonic calibration window (the weights never saw it, but the calibrator was fit on it); with K=3 Jan-Feb also use 2025-08/09, which are in-sample for the weights.
- Long-run rates for 2026 rows use all of 2025 (as backtest.py does), slightly more than BTS would have published at the time.
- 2026-01..08 is validation, not a holdout; the selection above used it and the clean holdout (2026-09 onward) was never loaded.
