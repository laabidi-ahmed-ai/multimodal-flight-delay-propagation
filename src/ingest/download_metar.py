"""Download raw METAR text (routine + special reports) from the Iowa Environmental Mesonet.

Usage:
    python -m src.ingest.download_metar --hubs-only     # 10 hubs (quick start)
    python -m src.ingest.download_metar                 # all stations in stations.csv

Files: <data_dir>/raw/metar/<IATA>.csv  with columns station, valid (UTC), metar
API docs: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def station_codes(cfg: dict, hubs_only: bool) -> list[str]:
    if hubs_only:
        return list(cfg["hubs"])
    path = Path(cfg["data_dir"]) / "raw" / "stations.csv"
    if not path.exists():
        raise SystemExit("stations.csv missing: run build_station_list first (or use --hubs-only).")
    return pd.read_csv(path)["iata"].tolist()   # IEM uses 3-letter FAA ids for US airports


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs-only", action="store_true")
    args = ap.parse_args()

    start, end = period(cfg, buffer_days=cfg.get("weather_buffer_days", 2))
    out_dir = raw_dir(cfg, "metar")
    for code in station_codes(cfg, args.hubs_only):
        out = out_dir / f"{code}.csv"
        if out.exists() and out.stat().st_size > 0:
            print(f"✓ {code} already downloaded")
            continue
        params = {
            "station": code, "data": "metar", "tz": "UTC", "format": "onlycomma",
            "report_type": "3,4",               # 3 = routine, 4 = specials (SPECI)
            "sts": f"{start:%Y-%m-%d}T00:00:00Z",
            "ets": f"{end:%Y-%m-%d}T23:59:00Z",
            "missing": "empty",
        }
        print(f"↓ METAR {code} {start} → {end}")
        r = http_get(URL, params=params)
        if r is None or not r.text.startswith("station"):
            print(f"   ! unexpected answer for {code}: {(r.text[:200] if r else '404')}")
        else:
            save_atomic(out, r.content)
            print(f"   saved {out.name} ({r.text.count(chr(10)) - 1} reports)")
        time.sleep(cfg.get("iem_sleep_seconds", 1.0))


if __name__ == "__main__":
    main()
