"""Download TAF forecasts from the Iowa Environmental Mesonet.

Usage:
    python -m src.ingest.download_taf --hubs-only
    python -m src.ingest.download_taf

Files: <data_dir>/raw/taf/<ICAO>.csv  (columns: station, valid = issue time, fx_valid,
       raw = raw TAF text, fx_end_valid, sknt, drct, gust, visibility, skyc, skyl, ..., ftype)

Parameters follow the "Automation Example with Curl" shown on
https://mesonet.agron.iastate.edu/request/taf.php :
    taf.py?station=...&tz=UTC&year1=..&month1=..&day1=..&hour1=0&minute1=0
          &year2=..&month2=..&day2=..&hour2=0&minute2=0&fmt=comma
The station id format (KATL vs ATL) is tried in both forms automatically.
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import airportsdata
import pandas as pd

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"


def stations(cfg: dict, hubs_only: bool) -> list[tuple[str, str]]:
    """Return (iata, icao) pairs."""
    if hubs_only:
        db = airportsdata.load("IATA")
        return [(c, db[c]["icao"]) for c in cfg["hubs"]]
    path = Path(cfg["data_dir"]) / "raw" / "stations.csv"
    if not path.exists():
        raise SystemExit("stations.csv missing: run build_station_list first (or use --hubs-only).")
    df = pd.read_csv(path)
    return list(zip(df["iata"], df["icao"], strict=True))


def has_rows(text: str) -> bool:
    """True if the answer is a CSV with a header containing 'raw' and at least one data row."""
    lines = text.strip().splitlines()
    return len(lines) > 1 and "raw" in lines[0]


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs-only", action="store_true")
    args = ap.parse_args()

    start, end = period(cfg, buffer_days=cfg.get("weather_buffer_days", 2))
    end_excl = end + timedelta(days=1)          # end bound is exclusive: include the whole last day
    base = {
        "tz": "UTC", "fmt": "comma",
        "year1": start.year, "month1": start.month, "day1": start.day, "hour1": 0, "minute1": 0,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "hour2": 0, "minute2": 0,
    }
    out_dir = raw_dir(cfg, "taf")
    for iata, icao in stations(cfg, args.hubs_only):
        out = out_dir / f"{icao}.csv"
        if out.exists() and out.stat().st_size > 0:
            print(f"✓ {icao} already downloaded")
            continue
        print(f"↓ TAF {icao} {start} → {end}")
        saved = False
        for code in (icao, iata):                # try 4-letter first, then 3-letter
            r = http_get(URL, params={**base, "station": code}, timeout=300)
            if r is not None and has_rows(r.text):
                save_atomic(out, r.content)
                print(f"   saved {out.name} via station={code} ({len(r.content) / 1e6:.1f} MB)")
                saved = True
                break
            time.sleep(cfg.get("iem_sleep_seconds", 1.0))
        if not saved:
            print(f"   ! no TAF rows for {icao}/{iata} (some small airports issue no TAF)")
        time.sleep(cfg.get("iem_sleep_seconds", 1.0))


if __name__ == "__main__":
    main()
