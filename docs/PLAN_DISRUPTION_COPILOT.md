# Plan: Phase 1 disruption copilot (pointer)

The canonical plan lives in the app repo, on its `feat/disruption-copilot` branch:

- https://github.com/porterpup/flyrely-app/blob/feat/disruption-copilot/docs/PLAN_DISRUPTION_COPILOT.md
- Local checkout: `../flyrely-app/docs/PLAN_DISRUPTION_COPILOT.md` (switch that repo to `feat/disruption-copilot`)

Edit only the canonical copy, so the two can't drift.

## API-side summary (see the plan for details)

- **Branch rules:** this branch (`feat/disruption-copilot`) is the integration branch for copilot work. Step PRs
  target it, never `master`. Merge `master` into it regularly.
- **Flag:** `FLYRELY_FEATURE_DISRUPTION_COPILOT=1` mounts `/copilot/*`. Unset (the default) means no routes, no
  `copilot` import and no watcher. Push needs `FLYRELY_COPILOT_PUSH=1` as well. Status source:
  `FLYRELY_COPILOT_STATUS_SOURCE=none|swim` (`swim` is a stub until roadmap R3).
- **New code goes in `copilot/`** (package), plus `scripts/build_route_schedule.py`, `scripts/gen_vapid_keys.py`,
  `models/route_schedule.csv.gz` and `migrations/003_copilot_push.sql`.
- **Allowed core edits:** about 4 lines at the end of `main.py` (flag-gated `copilot.mount(app, score=...)`) plus a
  small scorer helper, and `pywebpush` in `requirements.txt`. Nothing else in core files.
- **Never touch `backtest/` or `flyrely_features/`** (holdout freeze at 89be2a4).
- **Endpoints:** `GET /copilot/config`, `POST /copilot/options`, `POST /copilot/status`,
  `POST /copilot/push/subscribe`, `POST /copilot/push/unsubscribe`.
- **API steps:** S-A1 to S-A8. S-A1 to S-A5 can start now in parallel.
