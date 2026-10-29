"""Download raw METAR text (routine + special reports) from the Iowa Environmental Mesonet.

Usage:
    python -m src.ingest.download_metar --hubs-only     # 10 hubs (quick start)
    python -m src.ingest.download_metar                 # all stations in stations.csv

Files: <data_dir>/raw/metar/<IATA>.csv  with columns station, valid (UTC), metar
       <data_dir>/raw/metar/_station_ids_used.csv  (iata, id_used): IEM id that returned data
       <data_dir>/raw/metar/_no_data.txt  stations with no data (and the ids tried)
API docs: https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help

Station id: IEM does not always know an airport by its IATA code (Alaska/Hawaii/PR/VI need
the ICAO code, e.g. PANC; some airports have an FAA id different from the IATA code, e.g.
CLD -> CRQ; renamed airports, e.g. PBI -> DJT). For each station the script tries, in order:
weather_station_ids override (config.yaml) -> ICAO -> FAA id -> IATA, and keeps the first id
that returns at least one data row.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import airportsdata
import pandas as pd

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic, update_no_data

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


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


def candidate_ids(cfg: dict, iata: str, icao: str, db: dict) -> list[str]:
    """Ids to try, in order: config override -> ICAO -> FAA id -> IATA (duplicates removed)."""
    override = (cfg.get("weather_station_ids") or {}).get(iata, {}).get("metar", [])
    faa = (db.get(iata) or {}).get("lid")          # FAA location id, e.g. CLD -> CRQ
    ids = [*override, icao, faa, iata]
    return list(dict.fromkeys(i for i in ids if i))  # dict keeps the first occurrence, in order


def has_rows(text: str) -> bool:
    """True if the answer is a METAR CSV with at least one data row.

    IEM answers an unknown station with a valid CSV that has only the header line:
    that is "no data", not a success.
    """
    lines = text.strip().splitlines()
    return len(lines) > 1 and lines[0].startswith("station")


def update_ids_used(path: Path, ids_used: dict[str, str]) -> None:
    """Merge (iata, id_used) pairs into the CSV, keeping the rows of earlier runs."""
    old = (pd.read_csv(path, dtype=str) if path.exists()
           else pd.DataFrame(columns=["iata", "id_used"]))
    new = pd.DataFrame(sorted(ids_used.items()), columns=["iata", "id_used"])
    merged = pd.concat([old[~old["iata"].isin(new["iata"])], new]).sort_values("iata")
    merged.to_csv(path, index=False)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs-only", action="store_true")
    args = ap.parse_args()

    sleep = cfg.get("iem_sleep_seconds", 1.0)
    db = airportsdata.load("IATA")
    start, end = period(cfg, buffer_days=cfg.get("weather_buffer_days", 2))
    base = {
        "data": "metar", "tz": "UTC", "format": "onlycomma",
        "report_type": "3,4",               # 3 = routine, 4 = specials (SPECI)
        "sts": f"{start:%Y-%m-%d}T00:00:00Z",
        "ets": f"{end:%Y-%m-%d}T23:59:00Z",
        "missing": "empty",
    }
    out_dir = raw_dir(cfg, "metar")
    ids_path = out_dir / "_station_ids_used.csv"
    known_ids = (dict(pd.read_csv(ids_path, dtype=str).itertuples(index=False))
                 if ids_path.exists() else {})
    ids_used: dict[str, str] = {}
    failures: dict[str, str] = {}
    attempted: set[str] = set()
    for iata, icao in stations(cfg, args.hubs_only):
        out = out_dir / f"{iata}.csv"
        if out.exists() and out.stat().st_size > 0:
            print(f"✓ {iata} already downloaded")
            # Files saved before _station_ids_used.csv existed were all requested with the
            # IATA code (the old script used nothing else); record it if not listed yet.
            ids_used.setdefault(iata, known_ids.get(iata, iata))
            continue
        attempted.add(iata)
        tried = []
        for code in candidate_ids(cfg, iata, icao, db):
            print(f"↓ METAR {iata} via station={code} {start} → {end}")
            r = http_get(URL, params={**base, "station": code})
            time.sleep(sleep)                  # after EVERY request, failed ones included
            tried.append(code)
            if r is not None and has_rows(r.text):
                save_atomic(out, r.content)
                ids_used[iata] = code
                print(f"   saved {out.name} via station={code} "
                      f"({r.text.strip().count(chr(10))} reports)")
                break
            print(f"   no data rows for station={code}")
        else:                                  # loop ended without break: every id failed
            failures[iata] = ", ".join(tried)
            print(f"   ! no METAR for {iata} (tried {failures[iata]})")

    update_ids_used(ids_path, ids_used)
    update_no_data(out_dir / "_no_data.txt", attempted, failures)
    print(f"\nDone: {len(attempted) - len(failures)} downloaded, {len(failures)} without data "
          f"(see {out_dir / '_no_data.txt'}).")


if __name__ == "__main__":
    main()
