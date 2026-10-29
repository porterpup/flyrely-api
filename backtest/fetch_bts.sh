#!/bin/sh
# Download BTS On-Time Reporting monthly files (free, public) into data/bts/raw.
# Usage: sh backtest/fetch_bts.sh [first_year] [last_year] [last_month_of_last_year]
set -eu
FIRST=${1:-2024}; LAST=${2:-2026}; LAST_MONTH=${3:-8}
DIR="$(dirname "$0")/../data/bts/raw"
mkdir -p "$DIR"
for y in $(seq "$FIRST" "$LAST"); do
  for m in $(seq 1 12); do
    [ "$y" = "$LAST" ] && [ "$m" -gt "$LAST_MONTH" ] && continue
    f="On_Time_Reporting_Carrier_On_Time_Performance_1987_present_${y}_${m}.zip"
    if [ -s "$DIR/$f" ] && unzip -tq "$DIR/$f" >/dev/null 2>&1; then continue; fi
    echo "fetching $y-$m"
    curl -sf -m 900 -o "$DIR/$f" "https://transtats.bts.gov/PREZIP/$f"
    unzip -tq "$DIR/$f" >/dev/null
  done
done
