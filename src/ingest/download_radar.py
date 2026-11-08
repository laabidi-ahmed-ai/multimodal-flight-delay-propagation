"""Download national NEXRAD radar images hour by hour, crop around airports, keep only the crops.

National images are processed in memory and NOT saved (except the debug days listed in
config.yaml radar.keep_national_days), so the disk only holds small crops.

Usage:
    python -m src.ingest.download_radar --test-day 2025-07-15
        -> saves the national images of that day + crops for the 10 hubs (to inspect)
    python -m src.ingest.download_radar [--start YYYY-MM-DD] [--end YYYY-MM-DD]
        -> crops for the (airport, hour) pairs in radar_needed_hours.parquet
           (run src.ingest.radar_needed_hours first). Resumable and incremental: a day is
           skipped only if its .npz already holds every needed crop; otherwise only the
           missing hours are fetched and merged into it (e.g. after a new BTS month).
           Default range: period start - 1 day -> the day after the last BTS month on disk
           (radar is useless for days with no flight labels).

Output: <data_dir>/raw/radar_crops/YYYY/MM/YYYYMMDD.npz
        one compressed file per day; keys "<IATA>_<YYYYMMDDHH>" -> uint8 array (crop_px x crop_px)
        of raw palette values (convert to dBZ later, see IEM docs).
        Value 255 = outside the national image (crop near the edge): NOT "no rain" (0).
Coverage: the US composite covers the contiguous US only; airports outside it
(e.g. Alaska, Hawaii, Puerto Rico) are skipped and listed in skipped_airports.txt.
Missing server files are listed in missing_files.txt. Both lists are merged with earlier
runs, never overwritten.
Grid: every .wld read must match radar.expected_wld and every PNG must have the size
radar.expected_shape (config.yaml), otherwise the run stops.
Docs: https://mesonet.agron.iastate.edu/docs/nexrad_mosaic/
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic, to_date

BASE = "https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/GIS/uscomp/{name}"


OUTSIDE = 255   # fill value for crop pixels beyond the edge of the national image


def read_world_file(text: str, expected: list[float]) -> tuple[float, float, float, float]:
    """ESRI world file: A, D, B, E, C, F -> (pixel_w, pixel_h, lon_ul, lat_ul).

    Stops the run if the grid differs from `expected` (config radar.expected_wld): a changed
    grid would silently shift every crop.
    """
    values = [float(x) for x in text.split()]
    if len(values) != 6 or not np.allclose(values, expected):
        raise SystemExit(f"Radar grid changed: .wld = {values}, expected {expected}. "
                         "Check the IEM docs before cropping more days.")
    a, _d, _b, e, c, f = values
    return a, e, c, f


def crop(img: np.ndarray, lat: float, lon: float, wld, size: int) -> np.ndarray | None:
    """Square crop centred on (lat, lon). None if the airport is outside the image.

    Pixels beyond the image edge are filled with OUTSIDE (255), not 0: 0 means "no echo"
    (no rain), so padding with 0 would turn "no data" into "clear sky". In n0q, 255 would be
    95.5 dBZ, which does not occur in real weather, so it is safe to reserve it.
    """
    a, e, c, f = wld
    col = int(round((lon - c) / a))
    row = int(round((lat - f) / e))
    h, w = img.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    half = size // 2
    out = np.full((size, size), OUTSIDE, dtype=np.uint8)
    r0, r1, c0, c1 = row - half, row - half + size, col - half, col - half + size
    rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, h), max(c0, 0), min(c1, w)
    out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = img[rr0:rr1, cc0:cc1]
    return out


def to_array(png_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode != "P":
        print(f"   ! unexpected image mode {img.mode}; storing the first channel")
        arr = np.asarray(img)
        return arr[..., 0] if arr.ndim == 3 else arr
    return np.asarray(img)          # palette indices = radar intensity classes


def last_bts_day(raw: Path) -> date:
    """Last day of the most recent BTS month on disk (raw/bts/bts_YYYY_MM.zip)."""
    months = sorted(p.stem for p in (raw / "bts").glob("bts_*.zip"))   # bts_2026_08, ...
    if not months:
        raise SystemExit("No BTS files on disk: run download_bts first (or pass --end).")
    _, year, month = months[-1].split("_")
    first_of_next = (pd.Timestamp(int(year), int(month), 1) + pd.offsets.MonthBegin(1)).date()
    return first_of_next - timedelta(days=1)


def main() -> None:
    cfg = load_config()
    rcfg = cfg["radar"]
    product, size = rcfg["product"], rcfg["crop_px"]   # no default: the size must be explicit
    expected_shape = tuple(rcfg["expected_shape"])   # (rows, columns) of the national PNG
    keep_days = {to_date(d) for d in rcfg.get("keep_national_days", [])}
    sleep = cfg["iem_sleep_seconds"]   # IEM etiquette: <= 1 request/second

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-day", help="YYYY-MM-DD: hubs only, national images saved")
    ap.add_argument("--start", help="YYYY-MM-DD, first day (default: period start - 1 day)")
    ap.add_argument("--end", help="YYYY-MM-DD, last day (default: day after the last BTS month)")
    args = ap.parse_args()

    raw = Path(cfg["data_dir"]) / "raw"
    stations = pd.read_csv(raw / "stations.csv") if (raw / "stations.csv").exists() else None
    if stations is None:
        import airportsdata
        db = airportsdata.load("IATA")
        stations = pd.DataFrame([{"iata": c, "lat": db[c]["lat"], "lon": db[c]["lon"]}
                                 for c in cfg["hubs"]])
    coords = {r.iata: (r.lat, r.lon) for r in stations.itertuples()}

    if args.test_day:
        day = to_date(args.test_day)
        days = [day]
        keep_days.add(day)
        needed = {(h, datetime(day.year, day.month, day.day, hh))
                  for h in cfg["hubs"] for hh in range(24)}
    else:
        path = raw / "radar_needed_hours.parquet"
        if not path.exists():
            raise SystemExit("Run python -m src.ingest.radar_needed_hours first.")
        df = pd.read_parquet(path)
        df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True).dt.tz_localize(None)
        needed = set(zip(df["airport"], df["hour_utc"].dt.to_pydatetime(), strict=True))
        start, end = period(cfg, buffer_days=1)
        if args.start:
            start = to_date(args.start)
        # Radar is only useful for days that have flight labels: stop at the last BTS month on
        # disk, not today. The day after it is included because evening flights of the last
        # day have their T_pred hours after midnight UTC.
        end = to_date(args.end) if args.end else min(end, last_bts_day(raw) + timedelta(days=1))
        if end < start:
            raise SystemExit(f"Empty date range: {start} -> {end}.")
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    print(f"Radar days to process: {days[0]} -> {days[-1]} ({len(days)} days; "
          "days whose .npz already holds every needed crop are skipped)")

    by_hour: dict[datetime, list[str]] = {}
    for airport, t in needed:
        by_hour.setdefault(t, []).append(airport)

    crops_root = raw_dir(cfg, "radar_crops")
    skipped_path = crops_root / "skipped_airports.txt"
    missing_path = crops_root / "missing_files.txt"
    # Lists from earlier runs are kept and merged, never overwritten (no silent loss).
    skipped = read_lines(skipped_path)
    missing = read_lines(missing_path)
    n_missing_before = len(missing)
    n_new = n_updated = 0
    for day in tqdm(days, desc="days"):
        out = crops_root / f"{day:%Y/%m}" / f"{day:%Y%m%d}.npz"
        # A day is "done" only if its .npz holds every crop needed now. np.load(...).files reads
        # only the list of keys, not the arrays, so this check is cheap.
        have: set[str] = set()
        if out.exists():
            with np.load(out) as z:
                have = set(z.files)
                check_crop_size(out, z, size)
        todo = crops_to_fetch(day, by_hour, have, skipped, missing, product)
        if not todo:
            if not out.exists():                   # nothing needed this day: mark it done
                out.parent.mkdir(parents=True, exist_ok=True)
                save_npz(out, {})
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        keep = day in keep_days
        nat_dir = raw_dir(cfg, f"radar_national/{day:%Y/%m/%d}") if keep else None
        # This day is retried now: drop an earlier "no .wld at all" marker (added back below if
        # it fails again). Missing PNGs (404) stay listed: they do not exist on the server.
        missing.discard(f"{product}_{day:%Y%m%d}_all_wld")

        # Georeference: one .wld per hour exists; use the first available one of the day.
        wld = None
        for hh in range(24):
            wld_name = f"{product}_{day:%Y%m%d}{hh:02d}00.wld"
            r = http_get(BASE.format(t=day, name=wld_name))
            time.sleep(sleep)                      # after EVERY request, 404s included
            if r is not None:
                wld = read_world_file(r.text, rcfg["expected_wld"])
                if keep:
                    (nat_dir / wld_name).write_text(r.text)
                break
        if wld is None:
            # No .wld at all this day: no .npz is written, so the day is retried on restart.
            missing.add(f"{product}_{day:%Y%m%d}_all_wld")
            write_lines(missing_path, missing)
            continue

        new_crops: dict[str, np.ndarray] = {}
        for t, airports in sorted(todo.items()):  # only the hours that still lack a crop
            name = f"{product}_{t:%Y%m%d%H}00.png"
            r = http_get(BASE.format(t=t, name=name))
            time.sleep(sleep)                      # after EVERY request, 404s included
            # http_get returns None only on 404: the image does not exist on the server, so
            # retrying is useless; it is listed in missing_files.txt and the day is saved
            # without it.
            # Network/server errors raise after the retries instead: the run stops before this
            # day's .npz is written, so the missing hours are fetched again on restart.
            if r is None:
                missing.add(name)
                continue
            if keep:
                save_atomic(nat_dir / name, r.content)
            img = to_array(r.content)
            # The .wld only gives the top-left corner and pixel size: if the image itself had
            # another size, crops near the edges (and the 255 padding) would be wrong. Stop.
            if img.shape != expected_shape:
                raise SystemExit(f"Radar image {name} has shape {img.shape}, expected "
                                 f"{expected_shape} (radar.expected_shape). Check the IEM docs.")
            for ap_code in airports:
                if ap_code not in coords:
                    skipped.add(ap_code)
                    continue
                c = crop(img, *coords[ap_code], wld, size)
                if c is None:
                    skipped.add(ap_code)
                    continue
                new_crops[f"{ap_code}_{t:%Y%m%d%H}"] = c

        # Merge with the crops already saved for this day (e.g. hub hours saved before the
        # destination hours of a newly published BTS month were known).
        day_crops: dict[str, np.ndarray] = {}
        if out.exists():
            with np.load(out) as z:
                day_crops = {k: z[k] for k in z.files}
            n_updated += 1
        else:
            n_new += 1
        day_crops.update(new_crops)
        save_npz(out, day_crops)
        # Save the lists after every finished day, so a crash later keeps them in sync
        # with the .npz files already on disk.
        write_lines(skipped_path, skipped)
        write_lines(missing_path, missing)

    print(f"\nDone. {n_new} new days, {n_updated} days completed with missing crops. "
          f"{len(missing)} missing files ({len(missing) - n_missing_before:+d} this run), "
          f"{len(skipped)} airports outside coverage (lists in {crops_root}).")


def crops_to_fetch(day: date, by_hour: dict[datetime, list[str]], have: set[str],
                   skipped: set[str], missing: set[str], product: str
                   ) -> dict[datetime, list[str]]:
    """{hour: airports} still needed for `day` and not yet in its .npz (keys in `have`).

    Not counted as needed: hours whose PNG is listed in missing_files.txt (404 on the
    server, retrying is useless) and airports in skipped_airports.txt (outside the mosaic).
    """
    todo: dict[datetime, list[str]] = {}
    for hh in range(24):
        t = datetime(day.year, day.month, day.day, hh)
        if f"{product}_{t:%Y%m%d%H}00.png" in missing:
            continue
        airports = [a for a in by_hour.get(t, [])
                    if a not in skipped and f"{a}_{t:%Y%m%d%H}" not in have]
        if airports:
            todo[t] = airports
    return todo


def check_crop_size(path: Path, npz, size: int) -> None:
    """Stop if an existing .npz was made with another crop size (radar.crop_px changed).

    Crops of different sizes must never end up in the same file or dataset: the CNN expects
    one size, and the airport would not be at the same pixel. Loading ONE crop is enough,
    because a file is always written with a single crop size.
    """
    if not npz.files:                              # empty day file: nothing to compare
        return
    shape = npz[npz.files[0]].shape
    if shape != (size, size):
        raise SystemExit(f"{path} holds {shape[0]}x{shape[1]} crops but radar.crop_px = {size}. "
                         "Delete the old .npz files made with another crop size, then rerun.")


def save_npz(out: Path, crops: dict[str, np.ndarray]) -> None:
    """Write the day's crops atomically: a crash never leaves half a .npz behind."""
    tmp = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(tmp, **crops)
    tmp.replace(out)


def read_lines(path: Path) -> set[str]:
    """Non-empty lines of a text file, or an empty set if it does not exist."""
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def write_lines(path: Path, lines: set[str]) -> None:
    path.write_text("".join(f"{ln}\n" for ln in sorted(lines)), encoding="utf-8")


if __name__ == "__main__":
    main()
