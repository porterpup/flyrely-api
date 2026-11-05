#!/bin/sh
# Download BTS On-Time *Marketing Carrier* monthly files (free, public) into data/bts/marketing.
# These include all operating carriers (9E, PT, QX, G7, C5, YV are missing from the Reporting Carrier file).
# Usage: sh backtest/fetch_bts.sh [first_year] [last_year] [last_month_of_last_year]
#        FLYRELY_HOLDOUT_UNLOCK=YYYY-MM sh backtest/fetch_bts.sh --holdout YYYY-MM
#
# Holdout lock (mirrors backtest/holdout.py; HOLDOUT_START must equal holdout.LOCKED_FROM): months >= HOLDOUT_START
# are the untouched final test set. The normal form refuses them always. `--holdout YYYY-MM` downloads that one
# month, only into data/holdout/YYYY-MM/bts/, and only when FLYRELY_HOLDOUT_UNLOCK names the same month
# (the old value 1 is refused).
set -eu
HOLDOUT_START=2026-09
UNLOCKABLE="2026-09"   # months that may ever be unlocked (mirrors holdout.UNLOCKABLE)
HS_Y=${HOLDOUT_START%-*}; HS_M=$(expr "${HOLDOUT_START#*-}" + 0)

if [ "${1-}" = "--holdout" ]; then
  MONTH=${2-}
  if [ "$#" -ne 2 ]; then echo "usage: fetch_bts.sh --holdout YYYY-MM" >&2; exit 2; fi
  case $MONTH in
    [0-9][0-9][0-9][0-9]-0[1-9]|[0-9][0-9][0-9][0-9]-1[0-2]) ;;
    *) echo "invalid --holdout '$MONTH': expected YYYY-MM" >&2; exit 2;;
  esac
  HY=$(expr "${MONTH%-*}" + 0); HM=$(expr "${MONTH#*-}" + 0)
  if [ "$HY" -lt "$HS_Y" ] || { [ "$HY" -eq "$HS_Y" ] && [ "$HM" -lt "$HS_M" ]; }; then
    echo "--holdout $MONTH is not a protected month (the holdout starts at $HOLDOUT_START)" >&2; exit 2
  fi
  case " $UNLOCKABLE " in
    *" $MONTH "*) ;;
    *) echo "refusing --holdout $MONTH: only $UNLOCKABLE may be unlocked (later months stay locked until the policy is approved)" >&2; exit 2;;
  esac
  if [ "${FLYRELY_HOLDOUT_UNLOCK:-}" = "1" ]; then
    echo "FLYRELY_HOLDOUT_UNLOCK=1 is no longer accepted: the holdout unlock must name the month (FLYRELY_HOLDOUT_UNLOCK=$MONTH)" >&2
    exit 2
  fi
  if [ "${FLYRELY_HOLDOUT_UNLOCK:-}" != "$MONTH" ]; then
    echo "refusing --holdout $MONTH: the holdout is locked. Prefix this one command with FLYRELY_HOLDOUT_UNLOCK=$MONTH (never export it)" >&2
    exit 2
  fi
  DIR="$(dirname "$0")/../data/holdout/$MONTH/bts"
  mkdir -p "$DIR"
  f="On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_${HY}_${HM}.zip"
  if [ -s "$DIR/$f" ] && unzip -tq "$DIR/$f" >/dev/null 2>&1; then echo "already have $MONTH"; exit 0; fi
  echo "fetching holdout $MONTH"
  curl -sf -m 900 -o "$DIR/$f" "https://transtats.bts.gov/PREZIP/$f"
  unzip -tq "$DIR/$f" >/dev/null
  exit 0
fi

FIRST=${1-2024}; LAST=${2-2026}; LAST_MONTH=${3-8}   # an explicit empty arg is rejected below, not defaulted
DIR="$(dirname "$0")/../data/bts/marketing"

# Args must be purely numeric: a typo like "9x" must not slip past the lock below. POSIX sh only (runs under dash in CI).
for x in "$FIRST" "$LAST" "$LAST_MONTH"; do
  case $x in ''|*[!0-9]*) echo "invalid argument '$x': year/month arguments must be numeric" >&2; exit 2;; esac
done
# expr strips leading zeros portably ("09" -> 9) without bash-only 10# arithmetic.
LAST=$(expr "$LAST" + 0); FIRST=$(expr "$FIRST" + 0); LAST_MONTH=$(expr "$LAST_MONTH" + 0)
if [ "$LAST" -gt "$HS_Y" ] || { [ "$LAST" -eq "$HS_Y" ] && [ "$LAST_MONTH" -ge "$HS_M" ]; }; then
  echo "refusing to download ${LAST}_${LAST_MONTH}: months >= $HOLDOUT_START are the untouched holdout (use --holdout YYYY-MM with FLYRELY_HOLDOUT_UNLOCK=YYYY-MM)" >&2
  exit 2
fi

mkdir -p "$DIR"
for y in $(seq "$FIRST" "$LAST"); do
  for m in $(seq 1 12); do
    [ "$y" = "$LAST" ] && [ "$m" -gt "$LAST_MONTH" ] && continue
    f="On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_${y}_${m}.zip"
    if [ -s "$DIR/$f" ] && unzip -tq "$DIR/$f" >/dev/null 2>&1; then continue; fi
    echo "fetching $y-$m"
    curl -sf -m 900 -o "$DIR/$f" "https://transtats.bts.gov/PREZIP/$f"
    unzip -tq "$DIR/$f" >/dev/null
  done
done
