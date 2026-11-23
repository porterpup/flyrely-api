"""Amendment 1 re-test script: offset fit, window rule, freeze/check/score refusals, gate and C mappings. Offline:
no data files, no network, no models; the git calls are faked."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backtest"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import eval_arrival_connections as E  # noqa: E402
import eval_arrival_retest as R  # noqa: E402
from arrival_fakes import ARR_P, make_bundle  # noqa: E402
from flyrely_dist import ladder as L  # noqa: E402

RNG = np.random.default_rng(11)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ offset fit and the window rule

def test_fit_recovers_a_known_offset_with_slope_fixed_at_one():
    p = RNG.uniform(0.02, 0.9, 400_000)
    y = (RNG.random(len(p)) < sigmoid(R._logit(p) + 0.45)).astype(float)
    assert R.fit_offset(p, y) == pytest.approx(0.45, abs=0.02)
    y0 = (RNG.random(len(p)) < p).astype(float)                       # calibrated: no offset
    assert R.fit_offset(p, y0) == pytest.approx(0.0, abs=0.02)
    # the solution matches the mean of the labels exactly (score equation), and clipping keeps 0/1 finite
    a = R.fit_offset(np.r_[p, 0.0, 1.0], np.r_[y, 0.0, 1.0])
    assert np.isfinite(a)
    with pytest.raises(ValueError):
        R.fit_offset(p, np.zeros(len(p)))


def test_apply_offsets_adds_a_on_the_logit_scale_then_rearranges():
    off = {str(t): 0.3 for t in L.ARR_THRESHOLDS}
    P = np.tile(np.array(ARR_P), (3, 1))
    out = R.apply_offsets(P, off)
    assert np.allclose(R._logit(out[:, 3]), R._logit(P[:, 3]) + 0.3)
    assert (np.diff(out, axis=1) <= 1e-12).all()                      # still a survival curve
    off["15"] = -5.0                                                  # would break ordering at 15: sorted back
    out = R.apply_offsets(P, off)
    assert (np.diff(out, axis=1) <= 1e-12).all() and out.min() >= 0 and out.max() <= 1


def test_window_rule_is_the_trailing_six_months():
    assert R.fit_window("2026-09") == ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]
    assert R.fit_window("2027-01")[0] == "2026-07" and R.fit_window("2027-01")[-1] == "2026-12"
    assert R.EVAL_MONTH == "2026-09" and R.FIT_WINDOW_MONTHS == 6


class FakeStore:
    """meta/pred for the fit: records which months were requested."""

    def __init__(self, offset=0.35, n=3000):
        self.asked = []
        self.offset, self.n = offset, n

    def _month(self, ym):
        rng = np.random.default_rng(int(ym.replace("-", "")))
        P = np.sort(rng.uniform(0.01, 0.9, (self.n, len(L.ARR_THRESHOLDS))), axis=1)[:, ::-1]
        delay = rng.exponential(60, self.n) - 40
        return P.astype(np.float32), delay

    def meta(self, ym):
        self.asked.append(ym)
        _, d = self._month(ym)
        return pd.DataFrame({"cancelled": np.zeros(self.n, bool), "arr_delay": d})

    def pred(self, ym):
        self.asked.append(ym)
        return {"arr": self._month(ym)[0]}


def test_fit_offsets_reads_only_the_six_window_months():
    s = FakeStore()
    out = R.fit_offsets(s, R.fit_window())
    assert sorted(set(s.asked)) == R.fit_window()
    assert set(out) == {str(t) for t in L.ARR_THRESHOLDS}
    assert all(o["n"] == 6 * s.n and np.isfinite(o["a"]) for o in out.values())
    # offsets move the mean prediction to the window's late rate (ML score equation)
    assert all(abs(o["mean_pred_after"] - o["fit_window_late_rate"]) < 1e-3 for o in out.values())
    with pytest.raises(ValueError):
        R.fit_offsets(s, R.fit_window() + ["2026-09"])
    with pytest.raises(ValueError):
        R.fit_offsets(s, ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"])


def test_bl_table_json_roundtrip():
    tab = {"cell": {("ATL", 8): 0.1, ("ORD", 3): 0.2}, "bucket": {("lt60", 8): 0.15}, "all": 0.09}
    assert R.bl_table_from_json(json.loads(json.dumps(R.bl_table_to_json(tab)))) == tab


# ------------------------------------------------------------------ freeze

def test_freeze_refuses_when_a_september_arrival_file_exists(tmp_path):
    data = tmp_path / "data"
    (data / "bts" / "parquet_arr").mkdir(parents=True)
    (data / "cache" / "arrival").mkdir(parents=True)
    assert R.september_files(data) == []
    for rel in ("bts/parquet_arr/2026-09.parquet", "cache/arrival/2026-09.parquet", "cache/arrival/pairs_2026-09.parquet",
                "cache/arrival/pred_2026-09.npz"):
        f = data / rel
        f.write_bytes(b"x")
        assert [Path(x).name for x in R.september_files(data)] == [f.name]
        with pytest.raises(SystemExit, match="September|2026-09"):
            R.stage_freeze(tmp_path / "freeze.json", data, store=FakeStore())
        f.unlink()
    (data / "cache" / "arrival" / "2026-08.parquet").write_bytes(b"x")      # other months are fine
    assert R.september_files(data) == []


def test_freeze_refuses_unlock_existing_freeze_and_dirty_tree(tmp_path, monkeypatch):
    fp = tmp_path / "freeze.json"
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-10")
    with pytest.raises(SystemExit, match="never runs"):
        R.stage_freeze(fp, tmp_path, store=FakeStore())
    monkeypatch.delenv("FLYRELY_HOLDOUT_UNLOCK")
    fp.write_text("{}")
    with pytest.raises(SystemExit, match="written once"):
        R.stage_freeze(fp, tmp_path, store=FakeStore())
    fp.unlink()
    monkeypatch.setattr(R, "clean_tree", lambda: False)
    with pytest.raises(SystemExit, match="uncommitted"):
        R.stage_freeze(fp, tmp_path, store=FakeStore())
    assert not fp.exists()


# ------------------------------------------------------------------ score preflight

class FakeGit:
    def __init__(self):
        self.dirty, self.ancestor, self.changed = False, True, set()

    def __call__(self, *a):
        r = SimpleNamespace(returncode=0, stdout="")
        if a[0] == "status":
            r.stdout = " M x.py\n" if self.dirty else ""
        elif a[0] == "merge-base":
            r.returncode = 0 if self.ancestor else 1
        elif a[0] == "rev-parse":
            if a[1] == "HEAD":
                r.stdout = "HEADSHA\n"
            else:
                rev, path = a[1].split(":", 1)
                r.stdout = "new\n" if (rev == "HEAD" and path in self.changed) else "same\n"
        return r


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("FLYRELY_HOLDOUT_UNLOCK", raising=False)
    models, results, data = tmp_path / "models", tmp_path / "results", tmp_path / "data"
    models.mkdir(), results.mkdir()
    for k, f in R.MODEL_FILES.items():
        (models / f).write_bytes(k.encode())
    for k, p in R.retest_files(data).items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(("data-" + k).encode())
    freeze = {"eval_month": "2026-09", "code": {"commit": "FREEZESHA", "paths": R.CODE_PATHS},
              "models": R.hash_models(models)}
    fp = results / "arrival_retest_freeze.json"
    fp.write_text(json.dumps(freeze))
    (results / R.MARKER_NAME).write_text(json.dumps({"files": R.hash_data(data), "freeze_sha256": R.sha256_file(fp)}))
    git = FakeGit()
    kw = dict(results_dir=results, freeze_path=fp, data_root=data, models_dir=models, git=git)
    return SimpleNamespace(models=models, results=results, data=data, fp=fp, git=git, kw=kw, freeze=freeze)


def test_preflight_passes_on_a_good_setup(env):
    freeze, out = R.preflight_score(**env.kw)
    assert out == env.results / "arrival_retest_2026-09.json" and freeze["eval_month"] == "2026-09"


def test_score_refuses_without_the_freeze_file(env):
    env.fp.unlink()
    with pytest.raises(SystemExit, match="freeze"):
        R.preflight_score(**env.kw)


def test_score_refuses_without_the_check_marker(env):
    (env.results / R.MARKER_NAME).unlink()
    with pytest.raises(SystemExit, match="marker"):
        R.preflight_score(**env.kw)


def test_score_refuses_on_model_hash_mismatch(env):
    (env.models / "flight_delay_model.joblib").write_bytes(b"retrained")
    with pytest.raises(SystemExit, match="hash mismatch"):
        R.preflight_score(**env.kw)
    (env.models / "flight_delay_model.joblib").unlink()
    with pytest.raises(SystemExit, match="missing"):
        R.preflight_score(**env.kw)


def test_score_refuses_when_a_september_file_changed_since_check(env):
    R.retest_files(env.data)["pairs"].write_bytes(b"rebuilt")
    with pytest.raises(SystemExit, match="changed since"):
        R.preflight_score(**env.kw)


def test_score_refuses_when_the_freeze_file_changed_since_check(env):
    env.fp.write_text(json.dumps({**env.freeze, "created": "later"}))
    with pytest.raises(SystemExit, match="freeze file changed"):
        R.preflight_score(**env.kw)


def test_score_refuses_a_dirty_tree(env):
    env.git.dirty = True
    with pytest.raises(SystemExit, match="uncommitted"):
        R.preflight_score(**env.kw)


def test_score_refuses_code_that_differs_from_the_freeze_commit(env):
    env.git.changed = {"backtest/eval_arrival_retest.py"}
    with pytest.raises(SystemExit, match="differ from the freeze commit"):
        R.preflight_score(**env.kw)
    env.git.changed = {"flyrely_features"}
    with pytest.raises(SystemExit, match="differ"):
        R.preflight_score(**env.kw)
    env.git.changed = set()
    env.git.ancestor = False
    with pytest.raises(SystemExit, match="ancestor"):
        R.preflight_score(**env.kw)


def test_score_refuses_when_the_unlock_is_set(env, monkeypatch):
    monkeypatch.setenv("FLYRELY_HOLDOUT_UNLOCK", "2026-10")
    with pytest.raises(SystemExit, match="never runs"):
        R.preflight_score(**env.kw)


def test_the_retest_is_scored_once_and_rerun_rules(env):
    with pytest.raises(SystemExit, match="no earlier attempt"):
        R.preflight_score(force=True, reason="x", **env.kw)           # nothing to re-run yet
    freeze, out = R.preflight_score(**env.kw)
    R.record_attempt(out, freeze, results_dir=env.results, git=env.git)
    with pytest.raises(SystemExit, match="scored once"):
        R.preflight_score(**env.kw)                                   # second run refused
    with pytest.raises(SystemExit, match="--reason"):
        R.preflight_score(force=True, **env.kw)
    with pytest.raises(SystemExit, match="--reason"):
        R.preflight_score(force=True, reason="  ", **env.kw)
    _, out1 = R.preflight_score(force=True, reason="fixed a bug", **env.kw)
    assert out1.name == "arrival_retest_2026-09_rerun1.json"
    out.write_text("original")
    out1.write_text("rerun")
    _, out2 = R.preflight_score(force=True, reason="again", **env.kw)
    assert out2.name == "arrival_retest_2026-09_rerun2.json"          # never overwrites
    assert out.read_text() == "original" and out1.read_text() == "rerun"
    rec = json.loads((env.results / R.ATTEMPTS_NAME).read_text())
    assert len(rec) == 1 and rec[0]["freeze_commit"] == "FREEZESHA" and rec[0]["code_commit"] == "HEADSHA"


def test_attempt_is_recorded_before_any_outcome_is_loaded(env, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", env.results)
    monkeypatch.setattr(R, "FREEZE_PATH", env.fp)
    monkeypatch.setattr(R, "_git", env.git)
    monkeypatch.setattr(R, "preflight_score", lambda force, reason: (env.freeze, env.results / R.OUT_NAME))

    def boom():
        assert (env.results / R.ATTEMPTS_NAME).exists()               # already written when the data would load
        raise RuntimeError("stop before loading anything")

    monkeypatch.setattr(R, "make_store", boom)
    with pytest.raises(RuntimeError, match="stop before"):
        R.stage_score()
    assert len(json.loads((env.results / R.ATTEMPTS_NAME).read_text())) == 1


def test_main_flag_rules():
    with pytest.raises(SystemExit):
        R.main(["--stage", "check", "--force-rerun"])
    with pytest.raises(SystemExit):
        R.main(["--stage", "score", "--reason", "why"])


def test_score_never_fits():
    import inspect
    src = inspect.getsource(R.stage_score) + inspect.getsource(R.arrival_scoring) \
        + inspect.getsource(R.connection_confirmation)
    for fit in ("fit_offset", "fit_dependence", "fit_logistic_recal", ".fit(", "base_rate_table", "build_bl_table"):
        assert fit not in src


# ------------------------------------------------------------------ check stage

def test_check_passes_and_prints_no_outcome_then_fails_on_a_stray_date(tmp_path, monkeypatch, capsys):
    data, results = tmp_path / "data", tmp_path / "results"
    results.mkdir()
    files = R.retest_files(data)
    for p in files.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2026-09-01", "2026-09-30")
    v2 = pd.DataFrame({"FlightDate": days.repeat(2), "Origin": "ATL", "DepDelay": 5.0})
    v2.to_parquet(files["parquet_v2"])
    pd.concat([v2, pd.DataFrame({"ArrDelay": np.full(len(v2), 3.0)})], axis=1).to_parquet(files["parquet_arr"])
    pd.DataFrame({"ym": "2026-09", "flight_date": v2["FlightDate"]}).to_parquet(files["cache"])
    pd.DataFrame({"ym": "2026-09", "ym2": "2026-09", "arr_date": days, "miss": True}).to_parquet(files["pairs"])
    fp = results / "freeze.json"
    fp.write_text(json.dumps({"eval_month": "2026-09", "models": {}, "code": {"commit": "x"}}))
    monkeypatch.setattr(R, "EXPECTED_ROWS", len(v2))
    monkeypatch.setattr(R, "verify_models", lambda f, d=None: None)
    monkeypatch.setattr(R, "verify_frozen_code", lambda f, g=None: None)
    assert R.stage_check(fp, results, data) is True
    out = capsys.readouterr().out
    assert "ALL CHECKS PASSED" in out and "FAIL" not in out
    assert "miss" not in out and "delay" not in out.lower() and "rate" not in out
    marker = json.loads((results / R.MARKER_NAME).read_text())
    assert set(marker["files"]) == {"parquet_v2", "parquet_arr", "cache", "pairs"}
    assert marker["freeze_sha256"] == R.sha256_file(fp)
    # a date outside September fails and removes the earlier pass
    bad = v2.copy()
    bad.loc[0, "FlightDate"] = pd.Timestamp("2026-10-01")
    pd.concat([bad, pd.DataFrame({"ArrDelay": np.full(len(bad), 3.0)})], axis=1).to_parquet(files["parquet_arr"])
    assert R.stage_check(fp, results, data) is False
    assert "FAIL no date outside September" in capsys.readouterr().out
    assert not (results / R.MARKER_NAME).exists()


# ------------------------------------------------------------------ gate mapping (A1-A4 unchanged) and C mapping

def _rep(ece15=0.01, gap15=0.02, monthly=0.02, ece45=0.01, ece120=0.01, ratio=2.5, ordered=True, rps_lo=0.01):
    return {"thresholds": {"15": {"ece": ece15, "max_decile_gap": gap15}, "45": {"ece": ece45}, "120": {"ece": ece120}},
            "monthly_t15": {"2026-09": {"ece": monthly}},
            "tiers": {"high_over_low": ratio, "ordered_every_month": ordered},
            "rps": {"skill": {"point": 0.1, "lo": rps_lo, "hi": 0.2}}}


GOOD_CI, BAD_CI = {"point": -0.01, "lo": -0.02, "hi": -0.001}, {"point": -0.01, "lo": -0.02, "hi": 0.001}
COV = {"q10_q90": 0.80, "below_q50": 0.5}


def test_a1_thresholds_are_unchanged():
    ok = R.decide_arrival(_rep(), GOOD_CI, GOOD_CI, COV)
    assert ok["outcome"] == "ship_arrival_in_predict_and_headline" and all(ok[k]["pass"] for k in ("A1", "A2", "A3", "A4"))
    for kw in ({"ece15": 0.0201}, {"gap15": 0.0401}, {"monthly": 0.0401}, {"ece45": 0.0301}, {"ece120": 0.0301}):
        d = R.decide_arrival(_rep(**kw), GOOD_CI, GOOD_CI, COV)
        assert not d["A1"]["pass"] and d["outcome"] == "ship_nothing_arrival", kw
    edge = R.decide_arrival(_rep(ece15=0.02, gap15=0.04, monthly=0.04, ece45=0.03, ece120=0.03), GOOD_CI, GOOD_CI, COV)
    assert edge["A1"]["pass"]


def test_a2_a3_a4_and_the_outcome_mapping():
    d = R.decide_arrival(_rep(), BAD_CI, GOOD_CI, COV)                  # A2 fails vs production, A1 and A4 pass
    assert not d["A2"]["pass"] and d["A4"]["pass"] and d["outcome"] == "ladder_inside_connection_engine_only"
    assert not R.decide_arrival(_rep(), GOOD_CI, BAD_CI, COV)["A2"]["pass"]            # vs climatology too
    assert not R.decide_arrival(_rep(ratio=1.79), GOOD_CI, GOOD_CI, COV)["A3"]["pass"]
    assert R.decide_arrival(_rep(ratio=1.8), GOOD_CI, GOOD_CI, COV)["A3"]["pass"]
    assert not R.decide_arrival(_rep(ordered=False), GOOD_CI, GOOD_CI, COV)["A3"]["pass"]
    assert not R.decide_arrival(_rep(rps_lo=0.0), GOOD_CI, GOOD_CI, COV)["A4"]["pass"]
    assert not R.decide_arrival(_rep(), GOOD_CI, GOOD_CI, {"q10_q90": 0.7599})["A4"]["pass"]
    assert R.decide_arrival(_rep(), GOOD_CI, GOOD_CI, {"q10_q90": 0.84})["A4"]["pass"]
    # A1 passes, A3 fails, A2 passes: nothing shipped (other)
    assert R.decide_arrival(_rep(ratio=1.0), GOOD_CI, GOOD_CI, COV)["outcome"] == "ship_nothing_arrival_other"


def _c(c1=True, c2=True, c3=True, c4=True, c6=True):
    return {k: {"pass": v} for k, v in zip(("C1", "C2", "C3", "C4", "C6"), (c1, c2, c3, c4, c6))}


def test_c_confirmation_mapping_d4():
    assert R.confirmation_outcome(_c()) == E.COPULA_SHIP
    for k in ("c1", "c2", "c3"):
        assert R.confirmation_outcome(_c(**{k: False})) == "no_connection_numbers"            # no copula numbers
    assert R.confirmation_outcome(_c(c6=False)) == "no_connection_numbers_c6_failed"
    assert R.confirmation_outcome(_c(c4=False)) == "serve_layover_table_B-L_itin-layover-table-v1"
    assert R.confirmation_outcome(_c(c1=False, c4=False)) == "no_connection_numbers"
    # same ordering as the original connection_metrics: C6 failing does not hide the B-L fallback
    assert R.confirmation_outcome(_c(c4=False, c6=False)) == "serve_layover_table_B-L_itin-layover-table-v1"
    assert R.confirmation_outcome(_c(c1=False, c6=False)) == "no_connection_numbers"


def test_confirmation_outcome_matches_the_original_ordering_for_every_combination():
    import itertools
    for c1, c2, c3, c4, c6 in itertools.product([True, False], repeat=5):
        dec = _c(c1, c2, c3, c4, c6)
        orig = {k: {"pass": v["pass"]} for k, v in dec.items()}
        # the branch order of E.connection_metrics, evaluated on the same flags
        if c1 and c2 and c3 and c4 and c6:
            want = "ship_miss_probability_for_validated_connections"
        elif c1 and c2 and c3 and not c4:
            want = "serve_layover_table_B-L_itin-layover-table-v1"
        elif not (c1 and c2 and c3):
            want = "no_connection_numbers"
        else:
            want = "no_connection_numbers_c6_failed"
        assert R.confirmation_outcome(orig) == want


def test_freeze_refuses_model_hashes_that_differ_from_the_amendment():
    good = {k: {"file": "f", "sha256": v} for k, v in R.EXPECTED_MODEL_SHA256.items()}
    R.verify_expected_hashes(good)
    for k in good:
        bad = {**good, k: {"file": "f", "sha256": "0" * 64}}
        with pytest.raises(SystemExit, match=k):
            R.verify_expected_hashes(bad)
    with pytest.raises(SystemExit):
        R.verify_expected_hashes({})
    import re
    doc = (ROOT / "backtest" / "PROTOCOL_ARRIVAL_CONNECTIONS.md").read_text()
    for h in R.EXPECTED_MODEL_SHA256.values():
        assert h in doc                                                  # the constants are the amendment's values


def test_decide_arrival_is_the_same_code_as_the_original_block():
    """AST equality between decide_arrival and the A1-A4 block of eval_arrival_connections.arrival_evaluation."""
    import ast
    import inspect

    def body(fn_src, name):
        fn = next(n for n in ast.walk(ast.parse(fn_src)) if isinstance(n, ast.FunctionDef) and n.name == name)
        return fn.body

    def block(stmts):
        start = next(i for i, n in enumerate(stmts) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "t" and ast.unparse(n.value) == "rep['thresholds']")
        end = next(i for i, n in enumerate(stmts) if isinstance(n, ast.If) and i > start)
        return [ast.dump(n) for n in stmts[start:end + 1]]

    orig = block(body(inspect.getsource(E), "arrival_evaluation"))
    mine = block(body(inspect.getsource(R), "decide_arrival"))
    assert len(orig) == 7 and orig == mine


def test_clarification_16_uses_the_retest_arrival_outcome():
    assert R.arrival_passes("ship_arrival_in_predict_and_headline")
    assert R.arrival_passes("ladder_inside_connection_engine_only")
    assert not R.arrival_passes("ship_nothing_arrival") and not R.arrival_passes("ship_nothing_arrival_other")
    fin = R.final_connection_outcome
    assert fin("ship_arrival_in_predict_and_headline", E.COPULA_SHIP) == E.COPULA_SHIP
    assert fin("ladder_inside_connection_engine_only", E.COPULA_SHIP) == E.COPULA_SHIP
    # arrival failed: the C confirmation is not run and no copula numbers are served
    assert fin("ship_nothing_arrival", None) == "no_connection_numbers_arrival_ladder_not_shipped"
    assert fin("ship_nothing_arrival", E.COPULA_SHIP) == "no_connection_numbers_arrival_ladder_not_shipped"
    # non-copula outcomes are left as the C rules decide
    fb = "serve_layover_table_B-L_itin-layover-table-v1"
    assert fin("ship_nothing_arrival", fb) == fb and fin("ship_arrival_in_predict_and_headline", fb) == fb
    assert fin("ship_nothing_arrival", "no_connection_numbers") == "no_connection_numbers"


def test_ladder_bundle_fake_is_compatible_with_offsets():
    """The synthetic ladder's tail table works with the coverage function the score stage uses."""
    b = make_bundle()
    E.TAIL_ARR = b.arr_tail
    P = R.apply_offsets(np.tile(np.array(ARR_P), (50, 1)), {str(t): 0.2 for t in L.ARR_THRESHOLDS})
    cov = E.coverage(P, np.linspace(-20, 120, 50))
    assert 0 <= cov["q10_q90"] <= 1


def test_frozen_files_of_the_original_evaluation_are_not_in_the_retest_edit_set():
    assert "backtest/eval_arrival_connections.py" in R.CODE_PATHS and "backtest/holdout.py" in R.CODE_PATHS
