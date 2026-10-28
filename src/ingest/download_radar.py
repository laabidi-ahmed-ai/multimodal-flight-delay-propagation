"""Download national NEXRAD radar images hour by hour, crop around airports, keep only the crops.

National images are processed in memory and NOT saved (except the debug days listed in
config.yaml radar.keep_national_days), so the disk only holds small crops.

Usage:
    python -m src.ingest.download_radar --test-day 2025-07-15
        -> saves the national images of that day + crops for the 10 hubs (to inspect)
    python -m src.ingest.download_radar
        -> full period, crops for the (airport, hour) pairs in radar_needed_hours.parquet
           (run src.ingest.radar_needed_hours first). Resumable: finished days are skipped.

Output: <data_dir>/raw/radar_crops/YYYY/MM/YYYYMMDD.npz
        one compressed file per day; keys "<IATA>_<YYYYMMDDHH>" -> uint8 array (crop_px x crop_px)
        of raw palette values (convert to dBZ later, see IEM docs).
Coverage: the US composite covers the contiguous US only; airports outside it
(e.g. Alaska, Hawaii, Puerto Rico) are skipped and listed in skipped_airports.txt.
Docs: https://mesonet.agron.iastate.edu/docs/nexrad_mosaic/
"""
from __future__ import annotations

import argparse
import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic, to_date

BASE = "https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/GIS/uscomp/{name}"


def read_world_file(text: str) -> tuple[float, float, float, float]:
    """ESRI world file: A, D, B, E, C, F -> (pixel_w, pixel_h, lon_ul, lat_ul)."""
    a, _d, _b, e, c, f = (float(x) for x in text.split())
    return a, e, c, f


def crop(img: np.ndarray, lat: float, lon: float, wld, size: int) -> np.ndarray | None:
    """Square crop centred on (lat, lon), zero-padded at the edges. None if outside the image."""
    a, e, c, f = wld
    col = int(round((lon - c) / a))
    row = int(round((lat - f) / e))
    h, w = img.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    half = size // 2
    out = np.zeros((size, size), dtype=np.uint8)
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


def main() -> None:
    cfg = load_config()
    rcfg = cfg["radar"]
    product, size = rcfg["product"], rcfg.get("crop_px", 256)
    keep_days = {to_date(d) for d in rcfg.get("keep_national_days", [])}
    sleep = cfg["iem_sleep_seconds"]   # IEM etiquette: <= 1 request/second

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-day", help="YYYY-MM-DD: hubs only, national images saved")
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
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    by_hour: dict[datetime, list[str]] = {}
    for airport, t in needed:
        by_hour.setdefault(t, []).append(airport)

    crops_root = raw_dir(cfg, "radar_crops")
    skipped, missing = set(), []
    for day in tqdm(days, desc="days"):
        out = crops_root / f"{day:%Y/%m}" / f"{day:%Y%m%d}.npz"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        keep = day in keep_days
        nat_dir = raw_dir(cfg, f"radar_national/{day:%Y/%m/%d}") if keep else None

        wld_name = f"{product}_{day:%Y%m%d}0000.wld"
        r = http_get(BASE.format(t=day, name=wld_name))
        if r is None:
            missing.append(wld_name)
            continue
        wld = read_world_file(r.text)
        if keep:
            (nat_dir / wld_name).write_text(r.text)

        day_crops: dict[str, np.ndarray] = {}
        for hh in range(24):
            t = datetime(day.year, day.month, day.day, hh)
            airports = by_hour.get(t)
            if not airports:
                continue
            name = f"{product}_{t:%Y%m%d%H}00.png"
            r = http_get(BASE.format(t=t, name=name))
            # http_get returns None only on 404: the image does not exist on the server, so
            # retrying is useless; it is listed in missing_files.txt and the day is saved
            # without it.
            # Network/server errors raise after the retries instead: the run stops before this
            # day's .npz is written, so the day is redone from scratch on restart.
            if r is None:
                missing.append(name)
                continue
            if keep:
                save_atomic(nat_dir / name, r.content)
            img = to_array(r.content)
            for ap_code in airports:
                if ap_code not in coords:
                    skipped.add(ap_code)
                    continue
                c = crop(img, *coords[ap_code], wld, size)
                if c is None:
                    skipped.add(ap_code)
                    continue
                day_crops[f"{ap_code}_{t:%Y%m%d%H}"] = c
            time.sleep(sleep)

        tmp = out.with_name(out.stem + ".part.npz")
        np.savez_compressed(tmp, **day_crops)
        tmp.replace(out)

    (crops_root / "skipped_airports.txt").write_text("\n".join(sorted(skipped)), encoding="utf-8")
    (crops_root / "missing_files.txt").write_text("\n".join(missing), encoding="utf-8")
    print(f"\nDone. {len(missing)} missing files, {len(skipped)} airports outside coverage "
          f"(lists in {crops_root}).")


if __name__ == "__main__":
    main()
