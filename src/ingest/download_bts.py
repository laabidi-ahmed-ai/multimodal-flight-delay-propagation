"""Download BTS On-Time Performance monthly ZIP files (all US domestic flights).

Usage (from the repo root):
    python -m src.ingest.download_bts                       # whole period in config.yaml
    python -m src.ingest.download_bts --start 2025-07 --end 2025-07   # one month (test)

Files: <data_dir>/raw/bts/bts_YYYY_MM.zip  (kept as-is: Bronze layer)
Months not yet published by BTS are skipped and reported.
"""
from __future__ import annotations

import argparse

from src.ingest.utils import http_get, load_config, period, raw_dir, save_atomic

URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")


def month_range(start: str, end: str):
    """Yield (year, month) from 'YYYY-MM' to 'YYYY-MM' inclusive."""
    y, m = map(int, start.split("-")[:2])
    y2, m2 = map(int, end.split("-")[:2])
    while (y, m) <= (y2, m2):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main() -> None:
    cfg = load_config()
    p_start, p_end = period(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=f"{p_start:%Y-%m}")
    ap.add_argument("--end", default=f"{p_end:%Y-%m}")
    args = ap.parse_args()

    out_dir = raw_dir(cfg, "bts")
    missing = []
    for y, m in month_range(args.start, args.end):
        out = out_dir / f"bts_{y}_{m:02d}.zip"
        if out.exists() and out.stat().st_size > 0:
            print(f"✓ {out.name} already downloaded")
            continue
        print(f"↓ {y}-{m:02d} ...")
        r = http_get(URL.format(y=y, m=m))
        # BTS sometimes answers with an HTML page instead of a 404: check the ZIP signature.
        if r is None or not r.content.startswith(b"PK"):
            print(f"   – {y}-{m:02d} not available (probably not published yet)")
            missing.append(f"{y}-{m:02d}")
            continue
        save_atomic(out, r.content)
        print(f"   saved {out.name} ({len(r.content) / 1e6:.1f} MB)")

    print("\nDone.")
    if missing:
        print("Not available:", ", ".join(missing))


if __name__ == "__main__":
    main()
