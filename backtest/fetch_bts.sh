#!/bin/sh
# Download BTS On-Time *Marketing Carrier* monthly files (free, public) into data/bts/marketing.
# These include all operating carriers (9E, PT, QX, G7, C5, YV are missing from the Reporting Carrier file).
# Usage: sh backtest/fetch_bts.sh [first_year] [last_year] [last_month_of_last_year]
#
# Holdout lock: months >= HOLDOUT_START are the untouched final test set and are refused unless
# FLYRELY_HOLDOUT_UNLOCK=1.
set -eu
HOLDOUT_START=2026-09
FIRST=${1:-2024}; LAST=${2:-2026}; LAST_MONTH=${3:-8}
DIR="$(dirname "$0")/../data/bts/marketing"

# Args must be purely numeric: a typo like "9x" must not slip past the lock below. POSIX sh only (runs under dash in CI).
for x in "$FIRST" "$LAST" "$LAST_MONTH"; do
  case $x in ''|*[!0-9]*) echo "invalid argument '$x': year/month arguments must be numeric" >&2; exit 2;; esac
done
# expr strips leading zeros portably ("09" -> 9) without bash-only 10# arithmetic.
LAST=$(expr "$LAST" + 0); FIRST=$(expr "$FIRST" + 0); LAST_MONTH=$(expr "$LAST_MONTH" + 0)
HS_Y=${HOLDOUT_START%-*}; HS_M=$(expr "${HOLDOUT_START#*-}" + 0)
if [ "${FLYRELY_HOLDOUT_UNLOCK:-0}" != "1" ] && \
   { [ "$LAST" -gt "$HS_Y" ] || { [ "$LAST" -eq "$HS_Y" ] && [ "$LAST_MONTH" -ge "$HS_M" ]; }; }; then
  echo "refusing to download ${LAST}_${LAST_MONTH}: months >= $HOLDOUT_START are the untouched holdout (set FLYRELY_HOLDOUT_UNLOCK=1 to override)" >&2
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
