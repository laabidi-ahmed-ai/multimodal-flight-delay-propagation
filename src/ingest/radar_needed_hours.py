"""List which (airport, UTC hour) radar crops the flights need.

- Hubs: every hour of the period (many departures).
- Destinations: only the hours used by flights from the hubs:
  floor(T_pred) and the hours before it (frames_before_t_pred in config.yaml),
  where T_pred = scheduled departure (UTC) - t_pred_hours_before_departure.

Usage:  python -m src.ingest.radar_needed_hours
        python -m src.ingest.radar_needed_hours --month 2025-07 --dry-run   # quick check
Needs:  raw/bts/*.zip and raw/stations.csv
Output: <data_dir>/raw/radar_needed_hours.parquet  (airport, hour_utc)
        with --month: radar_needed_hours_<YYYY_MM>.parquet instead (never the full file)
Renamed destinations are mapped with airport_aliases (config.yaml), e.g. DJT -> PBI.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd

from src.ingest.utils import load_config, period


def read_bts(zip_path: Path, hubs: set[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(name) as f:
            df = pd.read_csv(f, usecols=["FlightDate", "Origin", "Dest", "CRSDepTime"],
                             dtype={"CRSDepTime": str}, low_memory=False)
    return df[df["Origin"].isin(hubs)]


def scheduled_departure_utc(df: pd.DataFrame, tz_by_iata: dict[str, str]) -> pd.Series:
    """Local scheduled departure -> UTC, handling '2400' and daylight-saving changes."""
    hhmm = df["CRSDepTime"].str.zfill(4)
    is_2400 = hhmm == "2400"
    hhmm = hhmm.where(~is_2400, "0000")
    local = pd.to_datetime(df["FlightDate"] + " " + hhmm, format="%Y-%m-%d %H%M", errors="coerce")
    local = local + pd.to_timedelta(is_2400.astype(int), unit="D")
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    n_unknown_tz = 0
    for origin, idx in df.groupby("Origin").groups.items():
        tz = tz_by_iata.get(origin)
        if tz is None:
            n_unknown_tz += len(idx)
            continue
        out.loc[idx] = (local.loc[idx]
                        .dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
                        .dt.tz_convert("UTC"))

    # Report every flight left without a UTC time (they are dropped later): nothing silent.
    # A local time that parsed, at an airport with a known time zone, can only become NaT
    # because it falls in the repeated hour of the autumn clock change (ambiguous="NaT").
    n_unparsed = int(local.isna().sum())
    n_clock_change = int(out.isna().sum()) - n_unparsed - n_unknown_tz
    if out.isna().any():
        print(f"   dropped (no UTC time): {n_clock_change} autumn clock-change hour, "
              f"{n_unparsed} unparseable date/time, {n_unknown_tz} unknown time zone")
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM: only this BTS month and its hub hours (test)")
    ap.add_argument("--dry-run", action="store_true", help="print the counts, write nothing")
    args = ap.parse_args()

    hubs = set(cfg["hubs"])
    aliases = cfg.get("airport_aliases") or {}
    raw = Path(cfg["data_dir"]) / "raw"
    stations = pd.read_csv(raw / "stations.csv")
    tz_by_iata = dict(zip(stations["iata"], stations["tz"], strict=True))
    offset = pd.Timedelta(hours=cfg["t_pred_hours_before_departure"])
    n_frames = cfg["radar"].get("frames_before_t_pred", 3)

    pattern = f"bts_{args.month.replace('-', '_')}.zip" if args.month else "bts_*.zip"
    zips = sorted((raw / "bts").glob(pattern))
    if not zips:
        raise SystemExit(f"No BTS file matches {pattern}.")

    parts = []
    for z in zips:
        df = read_bts(z, hubs)
        # Renamed airports (DJT -> PBI): use the stations.csv code, so the crop is cut around
        # the right coordinates instead of the destination being skipped as unknown.
        dest = df["Dest"].replace(aliases).reset_index(drop=True)
        dep_utc = scheduled_departure_utc(df, tz_by_iata)
        t_pred_hour = (dep_utc - offset).dt.floor("h")
        for k in range(n_frames):
            # Keep the Series (tz-aware UTC). `.values` would drop the time zone, giving naive
            # times that do not match the UTC hub rows below (object column, missed duplicates).
            parts.append(pd.DataFrame({
                "airport": dest,
                "hour_utc": (t_pred_hour - pd.Timedelta(hours=k)).reset_index(drop=True),
            }))
        print(f"{z.name}: {len(df):,} hub departures")

    dest_rows = pd.concat(parts).dropna().drop_duplicates()

    if args.month:
        start = pd.Timestamp(f"{args.month}-01", tz="UTC")
        last_hour = start + pd.offsets.MonthBegin(1) - pd.Timedelta(hours=1)
    else:
        start_day, end_day = period(cfg, buffer_days=1)
        start = pd.Timestamp(start_day, tz="UTC")
        last_hour = pd.Timestamp(end_day, tz="UTC") + pd.Timedelta(hours=23)
    all_hours = pd.date_range(start, last_hour, freq="h")
    hub_rows = pd.DataFrame([(h, t) for h in sorted(hubs) for t in all_hours],
                            columns=["airport", "hour_utc"])

    needed = (pd.concat([hub_rows, dest_rows]).drop_duplicates()
              .sort_values(["hour_utc", "airport"]))
    print(f"hour_utc dtype: {needed['hour_utc'].dtype}; "
          f"{len(hub_rows):,} hub rows + {len(dest_rows):,} destination rows "
          f"-> {len(needed):,} unique pairs")
    if args.dry_run:
        print("Dry run: nothing written.")
        return
    # A one-month list must never replace the full one: download_radar would then crop almost
    # nothing on the other days, save them as finished .npz files and skip them forever.
    # download_radar only reads radar_needed_hours.parquet, so the month file is for checks.
    name = (f"radar_needed_hours_{args.month.replace('-', '_')}.parquet" if args.month
            else "radar_needed_hours.parquet")
    out = raw / name
    needed.to_parquet(out, index=False)
    print(f"\nSaved {len(needed):,} (airport, hour) pairs "
          f"for {needed['airport'].nunique()} airports to {out}")


if __name__ == "__main__":
    main()
