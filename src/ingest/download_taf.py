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
Renamed airports listed under weather_station_ids in config.yaml (e.g. PBI: taf [KPBI, KDJT])
get one raw file per id; raw is never merged or modified (the merge happens in Silver).
Stations without TAF are listed in <data_dir>/raw/taf/_no_data.txt.
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import airportsdata
import pandas as pd

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic, update_no_data

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
    sleep = cfg.get("iem_sleep_seconds", 1.0)
    overrides = cfg.get("weather_station_ids") or {}
    out_dir = raw_dir(cfg, "taf")
    failures: dict[str, str] = {}
    attempted: set[str] = set()
    for iata, icao in stations(cfg, args.hubs_only):
        override = overrides.get(iata, {}).get("taf", [])
        if override:
            # Renamed airport: the IEM archive is split across several ids (e.g. KPBI then
            # KDJT). Download EACH id to its own raw file; they are merged later in Silver.
            for code in override:
                if fetch(code, [code], out_dir, base, sleep, start, end) is False:
                    failures[code] = f"{code} (override for {iata})"
                attempted.add(code)
            continue
        # Normal airport: one file named after the ICAO code; try 4-letter, then 3-letter.
        if fetch(icao, [icao, iata], out_dir, base, sleep, start, end) is False:
            failures[icao] = f"{icao}, {iata}"
        attempted.add(icao)

    update_no_data(out_dir / "_no_data.txt", attempted, failures)
    print(f"\nDone: {len(failures)} stations without TAF (see {out_dir / '_no_data.txt'}); "
          "some small or military airports issue no TAF.")


def fetch(name: str, codes: list[str], out_dir: Path, base: dict, sleep: float,
          start, end) -> bool | None:
    """Save raw/taf/<name>.csv from the first id in `codes` that returns TAF rows.

    Returns True if saved, False if no id returned rows, None if the file already existed.
    Sleeps after every request (failed ones included) to respect the IEM rate limit.
    """
    out = out_dir / f"{name}.csv"
    if out.exists() and out.stat().st_size > 0:
        print(f"✓ {name} already downloaded")
        return None
    print(f"↓ TAF {name} {start} → {end}")
    for code in codes:
        r = http_get(URL, params={**base, "station": code}, timeout=300)
        time.sleep(sleep)
        if r is not None and has_rows(r.text):
            save_atomic(out, r.content)
            print(f"   saved {out.name} via station={code} ({len(r.content) / 1e6:.1f} MB)")
            return True
        print(f"   no TAF rows for station={code}")
    print(f"   ! no TAF for {name} (tried {', '.join(codes)})")
    return False


if __name__ == "__main__":
    main()
