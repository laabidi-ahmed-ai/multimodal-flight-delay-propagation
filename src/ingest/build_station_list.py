"""Build the list of weather stations needed: the hubs + every destination served from them.

Reads the BTS ZIPs (Origin/Dest columns only) and maps each IATA code to its ICAO code,
coordinates and time zone with `airportsdata`.

Usage:  python -m src.ingest.build_station_list
Output: <data_dir>/raw/stations.csv  (iata, icao, lat, lon, tz, is_hub)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import airportsdata
import pandas as pd

from src.ingest.utils import load_config


def main() -> None:
    cfg = load_config()
    hubs = set(cfg["hubs"])
    bts_dir = Path(cfg["data_dir"]) / "raw" / "bts"
    zips = sorted(bts_dir.glob("bts_*.zip"))
    if not zips:
        raise SystemExit("No BTS files found. Run download_bts first.")

    airports = set(hubs)
    for z in zips:
        # BTS ZIPs may contain a readme next to the CSV: open the CSV member explicitly.
        with zipfile.ZipFile(z) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, usecols=["Origin", "Dest"], low_memory=False)
        airports |= set(df.loc[df["Origin"].isin(hubs), "Dest"].dropna().unique())
        print(f"read {z.name}: {len(airports)} airports so far")

    # Renamed airports: replace the new BTS code by the code airportsdata knows,
    # so one physical airport gives one station (e.g. DJT -> PBI).
    aliases = cfg.get("airport_aliases") or {}
    for new_code, known_code in aliases.items():
        if new_code in airports:
            print(f"alias: {new_code} -> {known_code}")
    airports = {aliases.get(code, code) for code in airports}

    db = airportsdata.load("IATA")
    rows, unknown = [], []
    for code in sorted(airports):
        info = db.get(code)
        if info is None or not info.get("icao"):
            unknown.append(code)
            continue
        rows.append({"iata": code, "icao": info["icao"], "lat": info["lat"],
                     "lon": info["lon"], "tz": info["tz"], "is_hub": code in hubs})

    out = Path(cfg["data_dir"]) / "raw" / "stations.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {len(rows)} stations to {out}")
    if unknown:
        print("No ICAO/time zone found for:", ", ".join(unknown), "(check manually)")


if __name__ == "__main__":
    main()
