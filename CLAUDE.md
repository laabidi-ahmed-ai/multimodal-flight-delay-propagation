# Multimodal Flight Delay & Propagation Prediction
**Project:** AeroGuard · Team CoreSix · Deep Learning Multimodal
**Axis:** Operational Efficiency (operational disruption risk)
**Duration:** 6 phases — deadline in 6 weeks. Phases are flexible: move on as soon as a phase's exit criteria are met.

> **This file is `CLAUDE.md`.** Claude Code loads it automatically in every session. It is the internal, detailed plan and rulebook. The public-facing presentation of the project is `README.md`.
>
> **Repo:** `laabidi-ahmed-ai/multimodal-flight-delay-propagation` (private for now; make public at the end of Phase 1–2 after checking no data/secrets are committed).

---

## 0. Current status and next steps

> **Keep this section updated after every step** (Claude Code updates it at the end of each task).
> Last update: 2026-11-07 (radar crop size 256 → 384 px).

**Phase:** 1 (Understand and connect the data). Step 1.1 setup done; downloads partly done.

**Done**
- Repo scaffold, venv, `config.yaml`, ruff config (passes).
- **BTS:** 32 months, 2024-01 → 2026-08 (2026-09/10 not published yet), 0.9 GB.
- **stations.csv:** 314 airports (10 hubs + destinations), Palm Beach once (`PBI, KDJT`).
- **METAR:** 314/314 airports with data (24 via a non-IATA id, see `_station_ids_used.csv`).
- **TAF:** 306/314 airports (+ Palm Beach as two files `KPBI` + `KDJT`); 8 without TAF.
- Radar scripts reviewed and fixed (grid + image-size checks, 255 padding, merged logs, sleeps, `--start/--end`, incremental resume that completes days missing crops, crop-size check on existing `.npz`).
- Radar test day 2025-07-15 run once at **256 px** (240 hub crops, 0.9 MB). **Crop size then changed to 384 px** → that file must be deleted by Ahmed and the test day redone.
- `docs/learning_log.md` template created.
- `notebooks/01_explore_bts.ipynb` **batch 1 written** (sections 0–6, July 2025): setup, counts, flight key, cancelled/diverted/tail, delay distribution, delay by hour/hub/airline, cause columns. Code tested outside Jupyter (runs; 631,428 rows ✅, key unique ✅, `DepDel15` check 0 mismatches ✅). **Not yet run by Ahmed; "My answer" cells empty.**

**Not started yet**
- Notebook 01 batch 2 (UTC times, one tail through a day, ground-time distribution → `chain_break_ground_time_hours`, consistency check, summary).
- Radar at 384 px (test day redo, `radar_needed_hours`, full run).
- Notebooks 02–03, tests, Silver/Gold, data-quality report, models.

**Next steps (in order)**
1. Delete the old 256 px test file `C:\data\flight-delay\raw\radar_crops\2025\07\20250715.npz`, then redo the radar test day at 384 px: `.venv\Scripts\python.exe -m src.ingest.download_radar --test-day 2025-07-15` (time it; inspect: ATL at pixel (192, 192), does 255 appear inside the real image, storms vs. METAR `TS`).
2. `.venv\Scripts\python.exe -m src.ingest.radar_needed_hours` (full).
3. Full radar run, overnight, resumable: `.venv\Scripts\python.exe -m src.ingest.download_radar` (or in slices with `--start/--end`). When a new BTS month is published: `download_bts` → `radar_needed_hours` → `download_radar` (completes the days that miss crops).
4. Meanwhile: run notebook 01 batch 1 and write the "My answer" cells; then ask for batch 2 (incl. ground-time distribution → `chain_break_ground_time_hours`, value chosen by Ahmed).

---

## Table of contents
0. [Current status and next steps](#0-current-status-and-next-steps)
1. [The objective](#1-the-objective)
2. [The final product: input → output](#2-the-final-product-input--output)
3. [The modalities](#3-the-modalities)
4. [Data sources](#4-data-sources)
5. [How the data links together](#5-how-the-data-links-together)
6. [Scope (deadline: 6 weeks)](#6-scope-deadline-6-weeks)
7. [Tech stack](#7-tech-stack)
8. [Repository structure](#8-repository-structure)
9. [Golden rules (non-negotiable)](#9-golden-rules-non-negotiable)
10. [The plan, phase by phase](#10-the-plan-phase-by-phase)
11. [Data tricks checklist](#11-data-tricks-checklist)
12. [Evaluation protocol](#12-evaluation-protocol)
13. [LLM layer rules](#13-llm-layer-rules)
14. [MLOps](#14-mlops)
15. [Questions a strict jury might ask (and answers)](#15-questions-a-strict-jury-might-ask-and-answers)
16. [Risks and fallbacks](#16-risks-and-fallbacks)
17. [Priorities: Must / Should / Could](#17-priorities-must--should--could)
18. [CV lines](#18-cv-lines)
19. [Glossary](#19-glossary)
20. [Learning resources](#20-learning-resources)
21. [Instructions for Claude Code](#21-instructions-for-claude-code)
22. [Related work and our contribution](#22-related-work-and-our-contribution)
23. [Decisions log](#23-decisions-log)

---

## 1. The objective

### In one sentence
Predict whether a flight will be delayed, and whether that delay will spread to the same aircraft's next flights, by fusing flight records, weather text, radar images and the aircraft's rotation chain, with a plain-language explanation for operations teams.

### Roadmap row
| Axis | Objective | What it brings | Technologies |
|---|---|---|---|
| Operational Efficiency | Multimodal prediction of delays and their propagation | Anticipates a flight's delay and its domino effect on the aircraft's next flights by combining flight data, weather and radar images, with a clear explanation for operations teams | Multimodal deep learning (cross-attention fusion), CNN on radar images, GRU/GNN on aircraft rotations, NLP/LLM (METAR/TAF, explanations), XGBoost (baseline) |

### Why delays spread (the core idea)
One aircraft flies several legs per day:

```
MIA → ATL (arrives 30 min late) → ATL → JFK (leaves late) → JFK → BOS (at risk)
```

A late inbound aircraft makes the next departure late. One delay becomes many. This is **delay propagation**.

### Geographic scope
- **United States only, domestic flights operated by US airlines** (BTS reporting carriers: Delta, American, United, Southwest, Alaska, JetBlue, Spirit, Frontier, regional partners such as SkyWest; check `Reporting_Airline` for the exact list per year).
- **Not included:** international flights (even by US airlines) and foreign airlines.
- Why: the US is the only source found with open, flight-by-flight delays **and** aircraft tail numbers. The method transfers to any airline with its own data (e.g. Tunisair).

### Why it matters
- Delays cost airlines money (crew, compensation, missed connections, fuel).
- It fills the **"Operational efficiency"** branch of the group mind map, which had no objective.
- It connects to the investigation report: failure propagation (Table 7, "Network disruption"), operational consequences, and the need for measured outcomes.

---

## 2. The final product: input → output

### Who uses it
An airline operations controller.

### Interface
1. Choose an **airport + date**.
2. See the list of flights coloured 🟢 on time / 🟠 risk / 🔴 likely large delay.
3. Click a flight to see the predicted delay, the risk to the aircraft's next flights, and an explanation.

### Model input (for one flight)
| Input | Example |
|---|---|
| Flight info | DL1234, ATL → JFK, scheduled 14:30 local |
| Previous flight of the same aircraft | MIA → ATL, departed 30 min late |
| Weather text (METAR/TAF) — **origin and destination** | ATL: "heavy thunderstorm, gusts 28 kt, low clouds"; JFK: TAF valid at arrival time |
| Radar images — **origin and destination** | crops around ATL and JFK at `T_pred` |

### Model output
```
Flight DL1234 (ATL → JFK)
Predicted departure delay: ~40 min
P(delay ≥ 15 min): 0.80

Propagation:
  → Next leg JFK → BOS: high risk (~25 min)

Explanation (LLM):
  "The aircraft is already running late from Miami and a thunderstorm
   is forecast over Atlanta at departure time. The next flight to Boston
   is also likely to be delayed."
```

### Prediction targets
1. **Departure delay in minutes** (regression)
2. **Delayed ≥ 15 min** yes/no (classification)
3. **Next-leg delay** of the same aircraft (propagation)

### Demo modes
| Mode | What it shows |
|---|---|
| **Replay** | A real stormy day in 2026: the model's predictions next to what actually happened. Proves it works. |
| **Live** | Today's real weather and radar plus a flight the user types in. Proves it is usable now. |

> Live mode needs the user to type the flight, because free real-time flight schedules and statuses are limited. A real airline would plug in its own feed.

---

## 3. The modalities

| # | Modality | Data type | Source | Encoder |
|---|---|---|---|---|
| 1 | Flight records | **Tabular** (numbers + categories) | BTS | MLP / XGBoost |
| 2 | Weather reports (origin + destination) | **Text** (coded) | METAR / TAF | Small pretrained language model (embeddings) |
| 3 | Radar crops (origin + destination) | **Image** (spatial; spatio-temporal if several frames) | IEM radar mosaics | CNN (ResNet18); ViT as an optional comparison |
| 4 | Aircraft rotation | **Sequence / graph** (relational + temporal) | Built from BTS (tail number) | GRU (GNN = stretch goal) |

**Encoders:** one encoder per modality — tabular (MLP or FT-Transformer), text, image, rotation — whose embeddings are fused with cross-attention.

**Core modalities to defend:** tabular + text + image. The rotation chain is the **structural** part of the model.

---

## 4. Data sources

All free. Verify each link works from your network in Phase 1 (setup).

| Dataset | Content | Link | Updated | Main fields |
|---|---|---|---|---|
| **BTS On-Time Performance** | One row per US domestic flight | Page: https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD · Fields: https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD · Monthly ZIP: `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip` | Monthly, published a few months later (**not live**) | `FlightDate`, `Reporting_Airline`, `Flight_Number_Reporting_Airline`, `Tail_Number`, `Origin`, `Dest`, `CRSDepTime`, `DepTime`, `CRSArrTime`, `ArrTime`, `DepDelay`, `ArrDelay`, `DepDel15`, `LateAircraftDelay`, `WeatherDelay`, `NASDelay`, `CarrierDelay`, `Cancelled`, `Diverted` |
| **METAR** (observed weather) | Hourly airport weather reports, raw text | Page: https://mesonet.agron.iastate.edu/request/download.phtml · Script: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help` | Near real time | station, time (UTC), raw METAR |
| **TAF** (forecasts) | 24–30 h airport forecasts, raw text, archive since 1996 | Page: https://mesonet.agron.iastate.edu/request/taf.php · Script: https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py?help | ~4 times/day | station, issue time, valid period, raw TAF |
| **Radar mosaics** | US radar image every 5 min | Docs: https://mesonet.agron.iastate.edu/docs/nexrad_mosaic/ · Archive: `https://mesonet.agron.iastate.edu/archive/data/YYYY/MM/DD/GIS/uscomp/` | Every 5 min | `n0r_YYYYMMDDHHMM.png` / `n0q_...png` + `.wld` (coordinates), UTC, EPSG:4326 |
| **airportsdata** (Python) | Airport lat/lon + time zone | https://pypi.org/project/airportsdata/ | Static | `iata`, `icao`, `lat`, `lon`, `tz` |
| *(optional)* FAA registry | Tail number → aircraft model / Mode S code | https://registry.faa.gov/database/ReleasableAircraft.zip | Daily | N-number, model, Mode S hex |

**TranStats reliability:** TranStats has had outages (e.g. a database recovery in June 2026). No evidence of blocking non-US visitors, but **test the download from Tunisia in Phase 1 (setup)**.

**Fallbacks if TranStats is unavailable:**
- Kaggle 2018–2022: https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022
- Kaggle 2023: https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft

**Known limits:**
- BTS cause columns are only filled for delays ≥ 15 min.
- IEM rate limits: about **1 request/second per IP**; a request **without specific stations is capped at 24 hours** of data; since January 2026 a single request is capped at **1,000 station-years**. Always query **station by station**.
- Radar `n0q` files are ~2–5 MB each; `n0r` ~0.3 MB. **Storage budget: ~120 GB free disk.** Used so far (2026-10-28): **~3.3 GB** (BTS 0.9 + METAR 1.0 + TAF 1.4; no radar yet). Expected at the end: **~34 GB of radar crops (384 px), ~40 GB in total**. `n0q` hourly for Jan 2024 → Oct 2026 (~25,000 images) would take ~45–115 GB as national images, too close to the limit. **Policy: keep `n0q` (high quality) but crop-and-delete:** download a month → cut crops for the needed airports/hours → delete that month's national images (keep **one** month of national images for debugging). Crops keep full native resolution, so quality is unchanged. **Test one day first** (download speed, crop, colour → dBZ).
- Crop size: **384 px** (384 px × 0.005° ≈ 213 km north–south, ≈ 178 km east–west at ATL (longitude degrees shrink with latitude)); the airport is at pixel **(192, 192)**. Decide once before the full run: you can shrink crops later, never enlarge them without re-downloading. `download_radar` stops if an existing `.npz` holds crops of another size (old files must be deleted, never mixed).
- Only crop what flights need: the 10 hubs at every hour, and destination airports at the hours used by flights (`T_pred` and the 2 hours before). Crops are cut **in memory while downloading**: national images are never saved (except `keep_national_days` in `config.yaml`), and crops are stored as **one compressed `.npz` per day**, not millions of small PNG files. The US composite covers the **contiguous US only**: Alaska/Hawaii/Puerto Rico destinations have no radar crop (listed in `skipped_airports.txt`; the model must handle a missing image).
- **Radar crop padding:** a crop that reaches beyond the edge of the national image is filled with **255** there, not 0. In `n0q`, 0 = no echo (no rain), so 0-padding would turn "no data" into "clear sky"; 255 would be 95.5 dBZ, which never occurs in real weather, so it is reserved for "outside the image". Mask `== 255` before converting to dBZ.
- **Radar grid check:** `download_radar` reads the day's first available hourly `.wld` (00:00, then 01:00, …) and stops if it differs from `radar.expected_wld` in `config.yaml` (n0q: 0.005° pixels, upper-left −126.0, 50.0; checked on 2024-01-01, 2025-07-15, 2026-10-27). It also stops if a national PNG is not `radar.expected_shape` = 5400 × 12200 pixels (27° / 0.005 × 61° / 0.005). `missing_files.txt` / `skipped_airports.txt` are merged with earlier runs and saved after each day.
- **Radar resume is incremental:** a day counts as done only if its `.npz` holds **every** crop key needed now (from `radar_needed_hours.parquet`), not counting PNGs listed in `missing_files.txt` (404 on the server) and airports in `skipped_airports.txt`. Otherwise the script fetches only the missing hours and saves the merged `.npz` atomically. Example: 2026-09-01 is first saved with hub hours only; when BTS September is published (rerun `radar_needed_hours`, then `download_radar`), its destination crops are added instead of the day being skipped forever. `radar_needed_hours --month` writes `radar_needed_hours_<YYYY_MM>.parquet`, never the full file (a one-month list would make `download_radar` save near-empty days as finished).
- Not available for free (do not plan on them): historical FAA ground stops / ground delay programs (public page keeps only ~14 days; full history in FAA ASPM, which requires an FAA account), NOTAM history (no clean free bulk archive), and any open flight-by-flight delay dataset for Tunisia or Europe comparable to BTS.
- Data is US-only. Say so explicitly; the method is general.

---

### Where the work happens (DuckDB + Databricks)
- **Raw downloads and the master copy of every table live on the PC** (`data_dir`, Parquet). Nothing is ever only in one tool.
- **Develop and debug the joins in DuckDB on the PC** (fast, no quota): write each transformation as a SQL file, test it on one month.
- **Run the real pipeline on Databricks:** upload the Bronze/Silver Parquet files, run the same SQL in Spark SQL, build the **Bronze → Silver → Gold Delta tables** there, and track experiments in **MLflow**. This is genuine project work (notebooks, Delta tables, job runs, MLflow screenshots), not a demo copy.
- **Export the Gold table back to the PC** as Parquet: it feeds Kaggle training and the demo.
- **If the Databricks daily quota stops you,** rerun the same SQL in DuckDB from the PC files and continue; nothing is lost.
- Keep SQL portable: where engines differ (e.g. `ASOF JOIN` in DuckDB vs. `ROW_NUMBER` in Spark), keep both variants side by side and check they give the same row counts.

### How to download (scripts in `src/ingest/`)
Run from the repo root, inside the virtual environment. All scripts read `config.yaml`, write to `<data_dir>/raw/`, skip files already downloaded (resumable) and print what is missing.

| Order | Command | Output |
|---|---|---|
| 1 | `python -m src.ingest.download_bts --start 2025-07 --end 2025-07` (test) then `python -m src.ingest.download_bts` | `raw/bts/bts_YYYY_MM.zip` |
| 2 | `python -m src.ingest.build_station_list` | `raw/stations.csv` (hubs + all destinations: IATA, ICAO, lat, lon, tz) |
| 3 | `python -m src.ingest.download_metar --hubs-only` (test) then without the flag | `raw/metar/<IATA>.csv` |
| 4 | `python -m src.ingest.download_taf --hubs-only` (test) then without the flag | `raw/taf/<ICAO>.csv` |
| 5 | `python -m src.ingest.download_radar --test-day 2025-07-15` (test: hubs only, national images kept to inspect) | `raw/radar_national/2025/07/15/` + `raw/radar_crops/2025/07/20250715.npz` |
| 6 | `python -m src.ingest.radar_needed_hours` | `raw/radar_needed_hours.parquet` (hubs every hour + destinations at the hours flights need) |
| 7 | `python -m src.ingest.download_radar` (full, overnight, resumable; optional `--start/--end YYYY-MM-DD` to download in slices; default end = day after the last BTS month on disk, not today) | `raw/radar_crops/YYYY/MM/YYYYMMDD.npz` — keys `<IATA>_<YYYYMMDDHH>`, 384×384 uint8 raw palette values |

Notes:
- METAR station ids on IEM are **not always the IATA code**. Alaska/Hawaii/Puerto Rico/Virgin Islands need the ICAO code (`PANC`, `PHNL`, `TJSJ`; `ANC` returns nothing). Airports whose FAA id differs from the IATA code need the FAA/ICAO id: AZA→IWA, CLD→CRQ, FCA→GPI, HHH→HXD, MQT→SAW, SCE→UNV, YUM→NYL. `download_metar` therefore tries, per station: `weather_station_ids` override (`config.yaml`) → ICAO → FAA id (`airportsdata` `lid`) → IATA, keeps the first id with data, and records it in `raw/metar/_station_ids_used.csv`. TAF uses 4-letter ICAO ids (`KATL`). ICAO codes come from `airportsdata`, never by prefixing `K`.
- BTS ZIPs may contain a readme next to the CSV: open the `.csv` member explicitly.
- TAF service parameters (from the IEM form's curl example): `year1, month1, day1, hour1, minute1, year2, …, tz=UTC, fmt=comma`; the script tries `KATL` then `ATL`. Some small airports issue no TAF: that is normal. Stations with no data are listed in `raw/metar/_no_data.txt` and `raw/taf/_no_data.txt`.
- Radar missing images: `http_get` returns `None` only on a 404 (the image does not exist on the server, so retrying is useless); it is listed in `missing_files.txt` and the day is saved without it. Network/server errors raise after the retries instead: the run stops before that day's `.npz` is written, so the day is redone from scratch on restart.

**Verified during setup (browser tests from Tunisia):** BTS ZIP ✅ (CSV ~26 MB compressed + `readme.html`), METAR ✅, TAF form ✅, radar PNG ✅.

**Raw-data gotchas seen in the files:**
- **Never open and save BTS CSVs in Excel.** Excel turns `"0659"` into `659` and breaks times. Read with pandas, and read time columns (`CRSDepTime`, `DepTime`, `CRSArrTime`, `ArrTime`, `WheelsOff`, …) **as strings**.
- `FlightDate` appears as `YYYY-MM-DD` in the data (the readme says `yyyymmdd`): parse explicitly and check.
- METAR: some reports are corrections (`COR`) → keep the corrected version, drop the original; a trailing `$` means the station needs maintenance (keep, but note it); specials (`SPECI`, e.g. at 20:46) appear between hourly reports.
- Thunderstorm info can appear in the remarks (`TS`, `CB`, `LTGICCG`, `TSB50`), not only in the main weather group: parse remarks too.
- TAF files have **one row per forecast group** (`FM`, `TEMPO`, …), not one per TAF; `valid` = issue time, `fx_valid`/`fx_valid_end` = period of that group. KATL shows ~12.6 distinct issue times/day (routine TAFs: 4/day): **check duplicates vs. amendments with `product_id` / `is_amendment`** before choosing "latest TAF issued ≤ `T_pred`". Read `is_amendment` with an **explicit dtype** (pandas reports mixed types).
- **Airport codes can change over time** (Palm Beach: `PBI` → `DJT` in BTS from Aug 2026, same `AirportID` 14027). Map renamed codes with `airport_aliases` in `config.yaml` (used by `build_station_list`), and in Silver identify airports by the stable BTS `OriginAirportID` / `DestAirportID`, not by the 3-letter code. `radar_needed_hours` applies the same aliases to `Dest`.
- **IEM can answer with a valid CSV that has only a header line** (unknown station id, no data). Treat it as **missing**, never as a successful download: the scripts do not save it and list the station in `_no_data.txt`.
- **The Palm Beach rename is handled differently per source on IEM:** the METAR history was **moved** to `DJT` (`PBI`/`KPBI` return nothing, even for 2025); the TAF archive is **split**: `KPBI` until ~2026-07-09, `KDJT` after. `weather_station_ids` in `config.yaml` (`PBI: {metar: [DJT], taf: [KPBI, KDJT]}`) handles it; the two TAF ids are kept as **two raw files** (`KPBI.csv`, `KDJT.csv`) and merged in Silver (dedupe the overlap, keep the original `station` column).
- **Incomplete weather coverage:** METAR for `WYS` (West Yellowstone, seasonal) starts 2024-04-29; `ROW` (Roswell) ends 2026-10-15 (station outage). **No TAF** for 8 airports: BLV, BQK, GRK, HHH (KHXD), MVY, SHD, VPS, YUM (KNYL), small or military/joint-use fields. The model must handle a missing TAF (like a missing radar crop).

## 5. How the data links together

Each source is like a separate notebook. To join two, you need a label present in both: the **key**.

```
                    ┌──────────────────────────────┐
                    │   FLIGHT ROW (BTS)            │
                    │   DL1234 · ATL→JFK · N123DL   │
                    └──────────────┬───────────────┘
        ┌──────────────────┬───────┴────────┬────────────────────┐
        │                  │                │                    │
 tail no. + UTC time  airport code + time   airport lat/lon + time   (optional) tail number
        │                  │                │                    │
        ▼                  ▼                ▼                    ▼
 previous / next     METAR + TAF       radar crop           FAA registry
 legs (rotation)     (ATL → KATL)      around ATL           (aircraft type)
```

| Link | Key | Rule |
|---|---|---|
| Flight → rotation | `Tail_Number`, ordered by UTC departure (across dates) | Same aircraft; new chain when the ground time exceeds the threshold chosen from the data (`chain_break_ground_time_hours`) or the link is broken |
| Flight → METAR | `Origin` → ICAO (`ATL` → `KATL`) + time | **Last report at or before the prediction moment** |
| Flight → TAF | ICAO + time | **Latest forecast issued at or before the prediction moment**, valid at departure time |
| Flight → radar (origin) | Origin lat/lon + time | Image at or just before `T_pred`, cropped around the origin |
| Flight → radar (destination) | Destination lat/lon + time | Image at or just before `T_pred`, cropped around the destination |
| Flight → destination weather | `Dest` → ICAO + time | Last METAR at or before `T_pred`; latest TAF issued at or before `T_pred`, **valid at scheduled arrival** |
| **Everything** | — | **All times converted to UTC first** |

**Rotation across midnight:** build chains per `Tail_Number` over **continuous UTC time** (not limited to one `FlightDate`), so late-evening flights and red-eyes stay linked to the next morning's leg when the ground time is short. Long ground times (overnight parking, maintenance, days off schedule) end the chain via the rule below. Useful features: `ground_time_before` (planned buffer), `hours_since_previous_arrival`, `is_first_leg_of_chain`.

**Broken-chain rule:** sort each tail number's legs by UTC departure. A link between two consecutive legs is valid only if (1) `next.Origin == previous.Dest`, (2) next departure is after previous arrival, and (3) the ground time is **below the threshold chosen from the data** (see "Aircraft ground time" below; not a fixed 12 h). Otherwise the chain is **broken** there (international leg missing from BTS, aircraft swap, data error, missing/blank tail number) → start a new chain segment, set `has_previous_leg = False` (previous-leg features empty), and exclude that pair from the next-leg (propagation) target. Log and report the share of broken links (expected small). Tests: `test_broken_chain_detection` (airport mismatch, long ground time, cancelled, diverted, overlapping legs, swap by actual timing).

**Aircraft ground time (overnight, maintenance, swaps):**
- Aircraft have no "rest days", but they park **overnight** (several hours) and leave service for **maintenance** (light checks often overnight; heavy checks for days or weeks). Airlines also **swap aircraft** when one is late or broken.
- Consequence: a delay mostly propagates **within the same operating day**; a long overnight stop usually absorbs it.
- **Choose the "new chain" threshold from the data** (Phase 1, step 1.2): plot the distribution of ground times between consecutive legs of the same tail. Expect two groups: short turnarounds (under ~2 h) and overnight stops (many hours). Put the break threshold between them instead of a fixed 12 h, and document the choice.
- Chain on tail number + ground-time gap (not only `FlightDate`), so late-evening flights that cross midnight are handled correctly.
- **Extra checks that use the flight/delay columns themselves:**
  - **Linking uses scheduled times** (`CRSDepTime`, `CRSArrTime`), which are known in advance, so building the chain itself causes no leakage.
  - **Cancelled leg** (`Cancelled = 1`): the plane did not fly it, so the next leg cannot inherit from it → break the chain at that leg (and never use a cancelled leg as a label).
  - **Diverted leg** (`Diverted = 1`): the plane ended at another airport → break the chain after it.
  - **Impossible actual timing** (historical cleaning only): if the next leg's actual departure is before the previous leg's actual arrival plus a minimum turnaround, the airline must have **swapped the aircraft** → break. This uses actual times, so it is used only to clean training/evaluation pairs, never as a model input; mention it in the report.
  - **Implausibly short scheduled turnaround** (e.g. under ~20 min) or **two legs of the same tail overlapping in time** → data error → break.
  - **Consistency check (analysis only, never an input):** on valid links, a next leg with `LateAircraftDelay > 0` should usually follow a late previous arrival. Report the share of links that agree; a low share means the chain logic needs fixing.
- Useful features: scheduled ground time (buffer), leg number of the day, first-flight-of-the-day flag, hours since the aircraft's first departure today.
- An aircraft missing for days (maintenance) simply starts a new chain when it returns.
- Note: a missing international round trip (ATL→JFK, [JFK→CDG→JFK missing], JFK→MIA) passes conditions 1 and 2 (same airport, later time); it is caught by **condition 3** because the hidden trip makes the ground time very long. Limitation: a very short missing hop (e.g. a quick cross-border round trip under the threshold) could slip through; this is rare and is mentioned in the report's limitations.

---

## 6. Scope (deadline: 6 weeks)

| Item | Choice |
|---|---|
| Airports | 10 hubs: ATL, ORD, DFW, DEN, LAX, JFK, CLT, LAS, PHX, SEA (flights departing from these) |
| Training | **2024 + 2025** (two full seasonal cycles). See "Why 2024 + 2025" below |
| Validation | First published months of 2026 (e.g. Jan–Feb) |
| Test | Remaining published months of 2026 (check BTS availability) |
| Radar | `n0q`, 1 image/hour, **384×384 crop** (`radar.crop_px` in `config.yaml`; 384 px × 0.005° ≈ 213 km north–south, ≈ 178 km east–west at ATL (longitude degrees shrink with latitude)) per airport, last 3 hourly frames. National images are **not** kept (crop in memory; one debug day kept). Models may downsample crops (e.g. to 128×128) at training time. |
| Data location | **Outside the repo and outside OneDrive**, e.g. `C:\data\flight-delay\`, read from `config.yaml` (`data_dir`) |
| Raw files | BTS, METAR, TAF **kept**. Radar national images **deleted after cropping** (one month kept for debugging) |
| Expected total size | ~40 GB in total (~34 GB of 384 px radar crops + BTS/METAR/TAF) with crop-and-delete (120 GB budget); never keep all national `n0q` images |
| Prediction moment | **2 hours before scheduled departure** |
| Graph | Rotation features + GRU; GNN only if time remains |
| ADS-B | Dropped (the 2024 adsb.lol archive alone is ~1,455 GiB; not needed for the core objective) |
| Deployment | Streamlit / Hugging Face Spaces; Azure only if time remains |

---

### Why 2024 + 2025 for training
- **Each season is seen twice.** Delays depend strongly on the season (summer thunderstorms, winter snow, holidays). With one year the model sees each month once; with two it learns what is typical, not what happened once.
- **More storm days.** Severe-weather days are rare; the radar and text modalities need many examples of them.
- **The years are similar.** Aeolus statistics: mean departure delay ≈ 12.2 min in 2023 and ≈ 12.6 min in 2024, so recent years behave alike and older data should not confuse the model.
- **Cost is small:** storage ~40 GB in total with radar crop-and-delete (120 GB budget), radar download about twice as long (overnight), same code.

### How to do it safely
1. **Download everything from the start** (`period.start: 2024-01-01` is already set in `config.yaml`).
2. **Develop and debug on 2025 only** (filter dates in notebooks; scripts take `--start/--end` where relevant). Small data = fast bugs fixing.
3. **Once the pipeline works, include 2024** in the Gold table and in training.
4. **Experiment (Phase 2):** train on 2025 only vs. 2024–2025 and compare on validation. If two years help → result for the report; if older data hurts → you have shown you checked for drift. Either way it is a point in your favour.

### Role of each period (who uses which data)

| Period | Role | Rule |
|---|---|---|
| 2024–2025 | **Training** | Model learns here |
| Early 2026 (e.g. Jan–Feb) | **Validation** | Tuning, early stopping, model choice |
| Rest of published 2026 | **Final test** | Touched **once** for the final numbers; never used to tune |
| 2026, month by month *(stretch)* | **Walk-forward replay** (MLOps demo) | Predict month M with a model trained up to M−1, then add M and retrain → every month is still predicted before it is learned from, so results stay honest |
| Months published after the project starts | **Monthly ingestion** (CI/CD) | The pipeline picks them up automatically: genuinely unseen data |
| Today's weather and radar | **Live demo** | No labels needed; flight entered by the user |
| A stormy 2026 day from the test period | **Replay demo** | Predicted vs. actual, side by side |

## 7. Tech stack

Rule: **every tool must solve a real problem in this project.** No logos for the sake of logos.

### ✅ Core (use these)
| Need | Tool | Cost | What it adds |
|---|---|---|---|
| Language | Python 3.11+ | Free | — |
| Local exploration and joins | pandas / Polars, **DuckDB**, Parquet | Free | Fast SQL on files, `ASOF`-style joins |
| Data platform (enterprise-style) | **Databricks Free Edition** (Spark SQL, Delta Lake) | Free forever (daily limits) | Medallion pipeline; Accenture's main data partner |
| Experiment tracking | **MLflow** (built into Databricks) | Free | Run comparison charts for the ablation |
| Data-quality checks | **pandera** (or Great Expectations) | Free | Automatic checks: UTC, no future features, no duplicates |
| Baseline | XGBoost, scikit-learn | Free | — |
| Deep learning | PyTorch, torchvision (ResNet18), Hugging Face (text embeddings) | Free | — |
| GPU training | **Kaggle** (main) / Google Colab | Free | Databricks Free Edition has no GPU: prepare data there or in DuckDB, train deep models on Kaggle |
| Explainability | **SHAP** | Free | "Why" plots per flight |
| LLM layer | An LLM API with a free tier, or a small open-source model (Hugging Face) | Free / free tier | Fact-checked explanations |
| API | FastAPI | Free | — |
| Demo app | **Streamlit + pydeck** | Free | **Interactive US map: flight chains turning red as delays propagate** (the key visual) |
| Demo hosting | **Hugging Face Spaces** | Free | Public link for CV/LinkedIn |
| Packaging | Docker | Free | — |
| CI/CD | **GitHub Actions** | Free (unlimited on public repos) | ✅ CI badge |
| Code quality | **ruff + pre-commit** | Free | Clean code, badge |
| Project management | **GitHub Projects** (kanban: Phase 1 … Phase 6) | Free | Visible planning |
| Diagrams | **Mermaid** (in README) + **Excalidraw** | Free | Architecture visuals |
| Weather parsing | A METAR parsing library (e.g. `python-metar`) | Free | — |
| Geo | `airportsdata`, Pillow / NumPy for radar crops | Free | — |

### 👍 Extras (Phases 5–6, only if time allows)
| Tool | Cost | Use |
|---|---|---|
| **Power BI Desktop** | Free | Business dashboard: delays by airport, airline, month, cause (consulting signal) |
| **Evidently AI** | Free (open source) | Drift reports during the monthly replay |
| **Azure for Students** (App Service / Container Apps) | $100/year credit, no card | Enterprise-style API deployment |
| **Codecov** | Free for public repos | Test-coverage badge |
| **Demo video** (OBS / Loom) | Free | 2-minute video in the README |
| Weights & Biases | Free personal/academic tier | Alternative to MLflow — pick **one**, not both |

### 🤔 Only with a real reason
| Tool | Verdict |
|---|---|
| Datadog (student pack) | Only if the API is deployed on Azure and you want latency/uptime monitoring |
| Azure DevOps | Duplicates GitHub Actions — use one |
| DVC | Data is re-downloadable by script — not needed |
| Airflow / Prefect | Overkill for one daily job — use GitHub Actions schedule or Databricks jobs |

### ❌ Skip
| Tool | Why |
|---|---|
| Vercel | For websites (Next.js); Streamlit on Hugging Face covers the demo |
| AWS | No natural role (data comes from BTS/IEM) |
| DigitalOcean | Student pack offer ended 31 July 2026 |
| Kubernetes | Far too heavy for one API |

### What recruiters will see
1. README: badges (CI, license, Python, coverage), Mermaid architecture, results table, demo GIF
2. Live demo link with the propagation map
3. 2-minute demo video
4. Screenshots: MLflow ablation, Databricks pipeline, Power BI dashboard
5. GitHub Projects board

---

## 8. Repository structure

```
flight-delay-multimodal/
├── CLAUDE.md                  # this file (or a short version of it)
├── README.md                  # final project presentation
├── pyproject.toml / requirements.txt
├── config.yaml                # data_dir, airports, date ranges, T_pred offset
├── .env                       # secrets (API keys) — NOT committed
│
│   DATA LIVES OUTSIDE THE REPO, at config.data_dir (e.g. C:\data\flight-delay\):
│   ├── raw/                   # Bronze: downloaded files as-is (bts/, metar/, taf/, stations.csv)
│   │   └── radar_crops/       # one compressed .npz per day: YYYY/MM/YYYYMMDD.npz,
│   │                          #   keys <IATA>_<YYYYMMDDHH> -> 384x384 uint8 palette values
│   ├── silver/                # cleaned, UTC, deduplicated
│   └── gold/                  # one row per flight, all features
├── notebooks/
│   ├── 01_explore_bts.ipynb
│   ├── 02_explore_weather_radar.ipynb
│   ├── 03_pilot_join.ipynb
│   ├── 04_baseline.ipynb
│   ├── 04b_training_period_comparison.ipynb
│   ├── 05_unimodal_models.ipynb
│   ├── 06_fusion_ablation.ipynb
│   └── 07_error_analysis.ipynb
├── src/
│   ├── ingest/                # download_bts.py, download_metar.py, download_taf.py, download_radar.py
│   ├── transform/             # to_utc.py, clean_bts.py, parse_metar.py, crop_radar.py, build_gold.py
│   ├── features/              # rotation.py, weather.py, traffic.py
│   ├── models/                # baseline.py, text_encoder.py, image_encoder.py, rotation_gru.py, fusion.py
│   ├── explain/               # shap_utils.py, llm_briefing.py
│   └── api/                   # main.py (FastAPI)
├── app/                       # streamlit_app.py
├── tests/                     # pytest tests (see section 10)
├── docs/
│   ├── learning_log.md
│   ├── data_quality_report.md
│   ├── results.md
│   └── architecture.png
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## 9. Golden rules (non-negotiable)

1. **All timestamps in UTC** before any join.
2. **No future information.** Every feature must be known at the prediction moment (scheduled departure − 2 h).
3. **Split by time, never randomly.** Train 2024–2025 → validate early 2026 → test later 2026. (Argued in the group's investigation report, §3.2.4 "Evaluation quality", and confirmed by Aeolus: random splits inflate AUC.)
4. **Fit preprocessing (scalers, encoders, vocabularies) on training data only.**
5. **Every join reports its match rate.** No silent drops.
6. **Every experiment is logged in MLflow** (params, metrics, data version).
7. **The LLM explains; it never predicts.** Every number it writes must match the model output.
8. **Report results honestly**, including where the model fails.
9. **Never commit raw data or secrets** (API keys) to git.
10. **Never keep code you cannot explain line by line.**

---

## 10. The plan, phase by phase

**Phases, not weeks.** Move to the next phase **as soon as** its exit criteria are met — finishing early is good; never skip exit criteria to go faster. Steps inside a phase are in recommended order.

| Phase | Content | Indicative pace (deadline: 6 weeks) |
|---|---|---|
| 1 | Understand and connect the data | ~1 week |
| 2 | Pipeline and baseline | ~1 week |
| 3 | One deep model per modality | ~1 week |
| 4 | Fusion and analysis (heart of the project) | ~1 week — protect this time |
| 5 | Explanations, LLM and demo | ~1 week |
| 6 | Finish and present | ~1 week, incl. buffer |

**Time guard:** if the total elapsed time is running far ahead of this pace, cut scope (section 17, Could → Should), never Phase 4.

Each phase has: **Tasks**, **Verify / tests**, **Must NOT happen**, **Check questions**, **Deliverable**.
Do not move on until you can answer the check questions without notes.

---

### PHASE 1: Understand and connect the data

#### Step 1.1: Setup
**Tasks**
- [ ] Get tutor approval for the objective and the "Operational Efficiency" axis
- [ ] Create GitHub repo with the structure from section 8; add `.gitignore` for `data/`
- [ ] Create accounts: Databricks Free Edition, Colab or Kaggle, (Azure for Students)
- [ ] Start `docs/learning_log.md`
- [ ] Create a **GitHub Projects** board with columns Phase 1 … Phase 6 and the tasks of this plan
- [ ] Set up **ruff** (config in `pyproject.toml`; pre-commit hooks only if the author installs them manually)
- [ ] Test that each data link in section 4 opens from your network (**especially TranStats from Tunisia**; switch to the Kaggle fallback if it fails)
- [ ] Download one BTS month (e.g. July 2025)

#### Step 1.2: Explore flight data (`01_explore_bts.ipynb`)
**Tasks**
- [ ] Count flights, airlines, airports
- [ ] Plot delay distribution (expect a long right tail)
- [ ] Share of cancelled / diverted flights
- [ ] Share of missing tail numbers
- [ ] Delays by hour of day, airport, airline
- [ ] Share of rows with cause columns filled (should match delays ≥ 15 min)
- [ ] Follow one aircraft (tail number) through a full day and watch delay accumulate

**Verify**
- [ ] Row count matches what BTS reports for that month (order of magnitude)
- [ ] `DepDel15 == (DepDelay >= 15)` for non-cancelled flights

**Check questions**
- Why do delays grow later in the day?
- What share of delay minutes is "late aircraft"?

#### Step 1.3: Weather text (`02_explore_weather_radar.ipynb`)
**Tasks**
- [ ] Download one day of METAR and TAF for KATL, KJFK, KMIA
- [ ] Decode 5 METARs by hand, then with a parsing library
- [ ] Extract: wind speed, gusts, visibility, cloud ceiling, thunderstorm flag, precipitation
- [ ] Read 3 TAFs; understand `FM`, `TEMPO`, `BECMG`, `AMD`
- [ ] **Start the radar download script in the background** (`n0q`, 1 image/hour, 2024 → latest 2026, crops only: test day first, then `radar_needed_hours`, then the full run)

**Check questions**
- What is the difference between METAR and TAF?
- Why must predictions 2 h ahead use TAF and not a METAR from departure time?

#### Step 1.4: Radar
**Tasks**
- [ ] Open one PNG and its `.wld` file
- [ ] Convert ATL lat/lon to pixel coordinates using the `.wld` values
- [ ] Crop 384×384 around ATL (already done by `download_radar --test-day`; inspect it: ATL at pixel (192, 192))
- [ ] Map colours/pixel values to storm intensity (dBZ)
- [ ] Display a crop during a known storm next to the matching METAR

**Verify**
- [ ] The airport is at the centre of the crop (plot a marker)
- [ ] A known storm day shows strong echoes; a clear day shows almost nothing

#### Step 1.5: Pilot join (`03_pilot_join.ipynb`)
**Tasks**
- [ ] Convert BTS local times → UTC with `airportsdata` time zones
- [ ] Handle `2400` and overnight arrivals
- [ ] For ONE flight: attach previous leg, last METAR before the prediction moment, latest TAF, radar crop
- [ ] Check every value by hand

**Tests to write** (`tests/test_time.py`, `tests/test_joins.py`)
- [ ] `test_utc_conversion`: ATL 14:30 local on 15 July = 18:30 UTC
- [ ] `test_utc_winter`: ATL 14:30 local on 15 January = 19:30 UTC
- [ ] `test_2400_handling`: `2400` becomes 00:00 the next day
- [ ] `test_overnight_arrival`: arrival before departure clock time is moved to the next day
- [ ] `test_metar_not_future`: matched METAR time ≤ prediction moment
- [ ] `test_taf_not_future`: matched TAF issue time ≤ prediction moment

**Must NOT happen**
- Weather from after the prediction moment attached to a flight
- Local and UTC times mixed in the same join
- A flight matched to another airport's weather

**Check question**
- List 3 ways a join could silently attach the wrong weather.

#### Step 1.6: Buffer + explain-it test

**Phase 1 exit criteria — move to the next phase as soon as all are true:**
- [ ] Deliverable done: exploration notebooks + one flight fully joined + passing time tests.
- [ ] Tests of this phase pass
- [ ] Check questions answered without notes (explain-it test)
- [ ] Learning log updated

---

### PHASE 2: Pipeline and baseline

#### Step 2.1: Medallion pipeline (Databricks or DuckDB locally, same logic)
**Tasks**
- [ ] **Bronze:** load raw BTS (2025 + 2026 months), METAR, TAF, radar file index
- [ ] **Silver:** UTC times, ICAO codes, cleaned tail numbers, deduplicated METARs (keep `SPECI`), latest TAF amendments, cancelled/diverted flagged
- [ ] **Gold:** one row per flight with all features and the radar crop path

**Data-quality checks:** define **pandera** schemas for Silver and Gold (types, allowed ranges, non-null keys, UTC timestamps).

**Tests** (`tests/test_gold.py`)
- [ ] `test_no_duplicate_flights`: (date, airline, flight number, origin) is unique
- [ ] `test_all_features_before_prediction_time`: every feature timestamp ≤ prediction moment
- [ ] `test_rotation_order`: legs of the same tail are ordered by UTC departure
- [ ] `test_radar_path_exists`: every referenced crop file exists

**Must NOT happen**
- Silent row loss in joins (always log counts before/after)
- Cancelled/diverted flights used as delay labels

#### Step 2.2: Data quality report (`docs/data_quality_report.md`)
- [ ] Rows per step (Bronze → Silver → Gold)
- [ ] Match rates: METAR, TAF, radar, previous leg
- [ ] Rows removed and why
- [ ] Missing-value table per feature

#### Step 2.3: Prediction moment + features
**Prediction moment:** `T_pred = scheduled departure (UTC) − 2 h`

**Features**
- [ ] **Previous leg:** if its actual departure ≤ `T_pred`, use its departure delay; otherwise mark as "not yet departed" and use only scheduled info
- [ ] **Planned buffer:** scheduled departure − previous leg's scheduled arrival
- [ ] Number of earlier legs today, cumulative delay known at `T_pred`
- [ ] Hour, weekday, month, holiday flag
- [ ] Airport traffic: scheduled departures in that hour
- [ ] Weather numbers from the last METAR and the TAF valid at departure (**origin**)
- [ ] Weather numbers from the last METAR and the TAF valid at scheduled arrival (**destination**)
- [ ] Radar crop paths for origin and destination at `T_pred`

**Must NOT happen (leakage traps)**
- Using the previous leg's **final arrival delay** if it lands after `T_pred`
- Using `LateAircraftDelay` / `WeatherDelay` / any cause column **as an input** (they are only known after the flight)
- Using the flight's own `DepTime`, `ArrTime`, `TaxiOut` as inputs

#### Step 2.4: Baselines (`04_baseline.ipynb`)
- [ ] Naive baseline: "same delay as known upstream delay at `T_pred`"
- [ ] XGBoost on tabular + parsed weather features
- [ ] Time split: train 2024–2025, validate early 2026, test later 2026
- [ ] Experiment: train on 2025 only vs. 2024–2025 → does more history help, or does older data hurt (drift)? Report it
  - Same code, **one parameterised function** called twice (no duplicated code)
  - Visible in `notebooks/04b_training_period_comparison.ipynb`: table + bar chart + 3-line conclusion
  - Both runs logged in MLflow; result added to `docs/results.md` and the README
- [ ] Metrics: MAE (minutes), ROC-AUC, precision, recall, F1 for ≥ 15 min
- [ ] Log in MLflow

**Tests**
- [ ] `test_split_no_overlap`: max(train date) < min(validation date) < min(test date)

**Check question**
- Why would a random split give fake good results?

**Phase 2 exit criteria — move to the next phase as soon as all are true:**
- [ ] Deliverable done: Gold table + data quality report + baseline results.
- [ ] Tests of this phase pass
- [ ] Check questions answered without notes (explain-it test)
- [ ] Learning log updated

> If Phase 2 is taking far longer than planned, cut to 5 airports. Never shorten Phase 4.

---

### PHASE 3: One deep model per modality (`05_unimodal_models.ipynb`)

#### Step 3.1: Tabular encoder
- [ ] MLP (or FT-Transformer) on the tabular features → embedding + prediction head
- [ ] Compare with XGBoost (deep tabular models often lose to trees — note it honestly)

#### Step 3.2: Text encoder
- [ ] Convert METAR/TAF to plain sentences ("gusts 28 knots, heavy thunderstorm, ceiling 1,200 ft")
- [ ] Encode with a small pretrained sentence-embedding model (frozen first)
- [ ] Small head → predict delay from text alone
- [ ] Compare: raw coded text vs. decoded sentences vs. parsed numbers

#### Step 3.3: Image encoder
- [ ] Pretrained ResNet18 on the crops (384×384, may be downsampled to 128×128 for training; frozen first, then fine-tune last block)
- [ ] Two crops per flight (origin + destination) through a **shared** CNN → two embeddings
- [ ] **Image-encoder comparison** (image-only model, same crops, same split, same epochs, selected on **validation**):

  | Candidate | Why |
  |---|---|
  | Small CNN from scratch (3–4 conv layers) | Reference: radar maps are not natural photos, ImageNet pretraining may help less |
  | **ResNet18** (pretrained) | Default starting point |
  | EfficientNet-B0 (pretrained) | Similar size, often more accurate |
  | ConvNeXt-Tiny (pretrained) | Modern CNN, heavier |
  | *(Optional)* small ViT | Only if time; needs more data |

  - Max 3–4 candidates (time budget: about 2 days)
  - Input trick: radar intensity is 1 channel → stack the **last 3 hourly frames as 3 channels** (time-as-channels), which fits pretrained 3-channel models
  - Log each run in MLflow (metrics, params, training time, model size)
  - The **winner** goes into the fusion model; the comparison table goes in `docs/results.md`
- [ ] Optional: stack the last 3 hourly frames (spatio-temporal)
- [ ] Predict delay from radar alone

#### Step 3.4: Rotation model
- [ ] Sequence of the aircraft's earlier legs today (delay known at `T_pred`, buffer, airports)
- [ ] GRU → embedding → prediction
- [ ] *(Stretch)* GNN over rotation graph with PyTorch Geometric

#### Step 3.5: Compare in MLflow

**Verify**
- [ ] Each unimodal model beats a trivial "predict the mean" baseline
- [ ] Training and validation curves look sane (no explosion, no perfect score)

**Must NOT happen**
- Suspiciously perfect scores (almost always leakage; go back to the Phase 2 tests)
- Normalisation statistics computed on validation/test data

**Check question**
- Which modality alone is strongest, and why?

**Phase 3 exit criteria — move to the next phase as soon as all are true:**
- [ ] Deliverable done: one score per modality, all logged.
- [ ] Tests of this phase pass
- [ ] Check questions answered without notes (explain-it test)
- [ ] Learning log updated

---

### PHASE 4: Fusion and analysis (`06_fusion_ablation.ipynb`, `07_error_analysis.ipynb`)

#### Step 4.1: Late fusion
- [ ] Combine the unimodal embeddings or outputs (concatenation + MLP)

#### Step 4.2: Cross-attention fusion
- [ ] Each modality = token(s); a small transformer/cross-attention block lets them interact
- [ ] Multi-task heads: delay minutes, P(≥15 min), next-leg delay

#### Step 4.3: Ablation table
Fill this table in `docs/results.md`:

| Model | MAE (min) | ROC-AUC | Recall@15 | Precision@15 | Calibration (ECE) | Storm-day MAE | Expected cost |
|---|---|---|---|---|---|---|---|
| Naive | | | | | | | |
| XGBoost (tabular + parsed weather) | | | | | | | |
| Fusion: all modalities | | | | | | | |
| Fusion: without text | | | | | | | |
| Fusion: without radar | | | | | | | |
| Fusion: without destination inputs | | | | | | | |
| Fusion: without rotation | | | | | | | |
| Fusion: tabular only | | | | | | | |

#### Step 4.4: Error analysis
- [ ] Storm days vs. clear days
- [ ] Hubs vs. each other
- [ ] Morning vs. evening
- [ ] Short vs. long buffers
- [ ] Worst 20 predictions: look at them one by one and explain why

**Verify**
- [ ] Every number in the table comes from the **test** period
- [ ] The same test set is used for every row

**Must NOT happen**
- Choosing the best model by looking at the test set (use validation)
- Hiding a result where fusion does not beat XGBoost; analyse it instead

**Check question**
- If radar adds little on average, where does it help and why?

**Phase 4 exit criteria — move to the next phase as soon as all are true:**
- [ ] Deliverable done: results table + honest conclusions. **This is the heart of the project.**
- [ ] Tests of this phase pass
- [ ] Check questions answered without notes (explain-it test)
- [ ] Learning log updated

---

### PHASE 5: Explanations, LLM and demo

#### Step 5.1: SHAP
- [ ] SHAP on XGBoost: global importance + per-flight top reasons

#### Step 5.2: LLM briefing (see section 13)
- [ ] Input: prediction, top reasons, decoded weather text
- [ ] Output: 2–4 sentence briefing
- [ ] Automatic check that every number in the text matches the model output

**Tests**
- [ ] `test_llm_numbers_match`: numbers in the briefing equal the prediction (± rounding)

#### Step 5.3: API (`src/api/main.py`)
- [ ] `POST /predict` → delay, probability, next-leg risk, explanation
- [ ] Input validation (airport code exists, time format)

**Tests**
- [ ] `test_api_predict`: returns 200 and the expected fields

#### Step 5.4: Streamlit demo (`app/streamlit_app.py`)
- [ ] **Replay mode:** pick a stormy 2026 day → predicted vs. actual
- [ ] **Live mode:** fetch today's METAR/TAF/radar; user types the flight
- [ ] Colour-coded flight list + detail view + explanation
- [ ] **pydeck map:** airports + flight legs; the selected aircraft's chain coloured by predicted delay (green → red)

#### Step 5.5: Docker + CI
- [ ] `Dockerfile` for API + app
- [ ] `.github/workflows/ci.yml` runs `ruff` + `pytest` on every push
- [ ] README badges: CI status, license, Python version (+ Codecov if used)

**Phase 5 exit criteria — move to the next phase as soon as all are true:**
- [ ] Deliverable done: working demo + passing CI.
- [ ] Tests of this phase pass
- [ ] Check questions answered without notes (explain-it test)
- [ ] Learning log updated

---

### PHASE 6: Finish and present

#### Step 6.1: Deploy
- [ ] Hugging Face Spaces (simplest) or Azure for Students
- [ ] *(Stretch)* monthly replay: feed 2026 month by month, retrain, champion vs. challenger, plot performance
- [ ] *(Extra)* **Evidently** drift report for the replay
- [ ] *(Extra)* **Power BI** dashboard (delays by airport, airline, month, cause); screenshot in README

#### Step 6.2: Documentation
- [ ] `README.md`: problem, architecture diagram, data sources, how to run, results table, limitations
- [ ] Report section for the group deliverable
- [ ] `docs/cloud_architecture.md`: short note (1 page + diagram) on how the system would run in production in the cloud — scheduled ingestion, Delta tables, model registry, containerised API, monitoring, monthly retraining

#### Step 6.3: Pitch
- [ ] 2-minute pitch: problem → data → model → results → demo → limits
- [ ] Record a 2-minute **demo video** (OBS / Loom) and add a GIF of the map to the README
- [ ] **LinkedIn post**: problem → approach → one key result → demo video/GIF → GitHub link (only once the repo is public)
- [ ] Architecture diagram (Mermaid in README + Excalidraw export in `docs/`)
- [ ] Rehearse the demo twice

#### Step 6.4: Buffer
Keep these days free. Something will break.

**Final test:** explain the whole project, every step, without notes.

---

## 11. Data tricks checklist

**Time**
- [ ] Everything in UTC; summer/winter time handled
- [ ] `2400` and overnight flights handled
- [ ] Features only from before `T_pred`
- [ ] Previous-leg info limited to what is known at `T_pred`

**Joins**
- [ ] Last METAR at or before `T_pred` (never after)
- [ ] Latest TAF issued at or before `T_pred`, valid at departure
- [ ] Match rates measured and reported
- [ ] Missing tail numbers, duplicates, cancellations handled
- [ ] `SPECI` (special) METARs kept; true duplicates removed

**Model training**
- [ ] Time split, never random
- [ ] Preprocessing fitted on training data only
- [ ] Extreme delays handled (log transform or capping for regression)
- [ ] Class imbalance handled (class weights, right metrics; not plain accuracy)
- [ ] Cause columns never used as inputs

**Honesty**
- [ ] Each modality's contribution measured (ablation)
- [ ] Failures reported

---

## 12. Evaluation protocol

| Item | Choice |
|---|---|
| Split | Train 2024–2025 · Validate early 2026 · Test later 2026 |
| Regression metric | MAE (minutes); also report on storm days |
| Classification metrics (≥ 15 min) | ROC-AUC, precision, recall, F1, calibration curve |
| Propagation metric | MAE on next-leg delay |
| **Cost-style metric** | Expected operational cost of the model's alerts: missed delays × cost of an unanticipated delay + false alarms × cost of an unnecessary action. Cost values live in `config.yaml` (`cost_missed_delay`, `cost_false_alarm`) and are **stated assumptions with a cited source or a sensitivity analysis** — never presented as real airline figures |
| Baselines | Naive upstream delay; XGBoost |
| Model selection | On validation only |
| Ablation | Remove one modality at a time (section 10, Phase 4) |
| Error analysis | Storm/clear, hub, time of day, buffer length, worst 20 cases |

> Accuracy alone is misleading: most flights are on time, so a model that always says "on time" looks accurate. Always report precision and recall.

---

## 13. LLM layer rules

**Role:** turn the model's prediction into a short, clear explanation. **Never** let the LLM make the prediction.

**Input given to the LLM (structured only)**
```json
{
  "flight": "DL1234 ATL-JFK",
  "predicted_delay_min": 40,
  "p_delay_15": 0.80,
  "next_leg": {"route": "JFK-BOS", "predicted_delay_min": 25},
  "top_reasons": ["inbound aircraft 30 min late", "thunderstorm forecast at departure", "evening peak traffic"],
  "weather_text": "heavy thunderstorm, gusts 28 knots, ceiling 1,200 ft"
}
```

**Rules**
- Use only the given facts; no outside claims
- 2–4 sentences, plain language
- Every number must match the input (automatic test)
- If confidence is low, say so

**Also usable as the text encoder:** decode METAR/TAF into sentences, then embed (Phase 3).

---

## 14. MLOps

| Piece | Minimum (Must) | Better (Should/Could) |
|---|---|---|
| Tracking | MLflow for every run | Model registry with versions |
| Tests | pytest for time, joins, leakage, API | Data validation suite |
| CI | GitHub Actions runs tests on push | Lint + small training smoke test |
| Packaging | Dockerfile | Separate API / app images |
| Deployment | Hugging Face Spaces | Azure for Students |
| Data refresh | Daily METAR/TAF/radar ingestion script | Monthly BTS retraining, champion vs. challenger, replayed over 2026 |

> Honest limit: BTS labels arrive monthly with a lag, so retraining can only be monthly, not daily.

---

## 15. Questions a strict jury might ask (and answers)

### Weak point 1: "METAR is not real text; you can parse it into numbers."
**Answer:** It is coded text, so we tested both. We compare (a) hand-parsed numeric features, (b) raw coded text embeddings, and (c) decoded plain-language sentence embeddings. The result shows whether a language-model representation adds value, which justifies how we treat this modality either way.

### Weak point 2: "The rotation chain comes from the same flight table; it is not a separate source."
**Answer:** Correct: it is a different **view** (relational/temporal) of the same data, not a separate sensor. We present tabular + text + image as the three core modalities, and the rotation chain as the structural component that models propagation. Its contribution is measured separately in the ablation.

### Other likely questions
| Question | Answer |
|---|---|
| "Does fusion beat XGBoost?" | Show the ablation table. If not overall, show where it helps (e.g. storm days) and explain why. |
| "How do you know there is no leakage?" | Time split, `T_pred` rule, cause columns excluded, and automated tests (section 10). |
| "Why US data, not Tunisia?" | It is the only open, flight-level delay dataset with tail numbers. The method transfers to any airline with its own data. |
| "Why is this aviation *risk* prevention?" | It addresses **operational disruption risk** and the propagation chain identified in the investigation (Table 7, "Network disruption"). |
| "Hasn't this been done (Aeolus, NeurIPS 2025)?" | Aeolus provides tabular + flight-chain + graph data with hourly weather numbers, and builds chains from flight numbers. We add radar images and METAR/TAF text through a language model, build chains from real tail numbers, use a strict prediction moment with forecasts, and deliver an explainable, deployed system (see section 22). |
| "Can it run live?" | Weather and radar are live; flight labels are monthly. Live mode uses today's weather with a user-entered flight; replay mode proves accuracy. |
| "Why an LLM?" | To make predictions understandable for operations staff; it never predicts, and its numbers are checked automatically. |

---

## 16. Risks and fallbacks

| Risk | Sign | Fallback |
|---|---|---|
| TranStats unreachable | Download fails during setup | Kaggle mirrors (section 4) |
| Pipeline too slow | Phase 2 much longer than planned | Cut to 5 airports |
| Radar download too slow | Not finished when Phase 3 starts | Fewer hours (e.g. every 2 h) or fewer months |
| Radar adds nothing | Ablation shows ~0 gain | Report it honestly; focus on storm-day analysis and rotation |
| Fusion worse than XGBoost | Phase 4 table | Analyse why; keep XGBoost as production model, fusion as research result |
| Free GPU limits | Colab disconnects | Smaller crops, frozen encoders, Kaggle GPU |
| LLM hallucination | Numbers mismatch | Template-based fallback text |
| Time pressure | Behind schedule | Drop Could → Should items, never Must items |

---

## 17. Priorities: Must / Should / Could

**Must**
- [ ] Clean joined Gold dataset
- [ ] XGBoost baseline
- [ ] Multimodal fusion model + ablation table
- [ ] LLM explanation
- [ ] Demo (replay + live)
- [ ] README + report

**Should**
- [ ] Cross-attention fusion
- [ ] Destination weather + radar
- [ ] Cost-style metric
- [ ] Cloud architecture note
- [ ] Demo video + LinkedIn post
- [ ] SHAP
- [ ] GitHub Actions CI
- [ ] Docker

**Could**
- [ ] GNN on rotations
- [ ] Azure deployment
- [ ] Monthly retraining replay
- [ ] Databricks certification (Data Engineer Associate)

---

## 18. CV lines

> *Built a multimodal deep-learning system (flight records, weather text, radar imagery, aircraft-rotation sequences) predicting flight delays and their propagation; cross-attention fusion improved MAE by X% over an XGBoost baseline on 2026 test data; LLM-generated operational briefings with automated factual checks.*

> *Designed a multi-source data pipeline on Databricks (Spark SQL, Delta Lake, medallion architecture) with leakage-safe temporal joins and automated data-quality tests.*

> *Deployed the model as a FastAPI service with a Streamlit demo, Docker, MLflow tracking and CI/CD on GitHub Actions.*

(Replace X with your real result.)

---

## 19. Glossary

| Term | Meaning |
|---|---|
| **ATC** | Air Traffic Control (not used in this objective; the previous objective) |
| **BTS** | US Bureau of Transportation Statistics |
| **Tail number** | The aircraft's registration, e.g. N123DL |
| **Rotation** | The chain of flights one aircraft flies in a day |
| **Turnaround / buffer** | Time on the ground between two legs |
| **Propagation** | A delay passing from one flight to the next |
| **METAR** | Routine airport weather observation (coded text) |
| **TAF** | Airport weather forecast (coded text) |
| **SPECI** | Special METAR issued when weather changes fast |
| **dBZ** | Radar reflectivity; higher = stronger precipitation/storm |
| **UTC** | Universal time used for all joins |
| **`T_pred`** | Prediction moment = scheduled departure − 2 h |
| **Leakage** | Using information not available at prediction time |
| **Ablation** | Removing one part to measure its contribution |
| **Medallion** | Bronze (raw) → Silver (clean) → Gold (ready for ML) |
| **MAE** | Mean absolute error, in minutes |
| **ROC-AUC** | How well the model ranks delayed vs. on-time flights |

---

## 20. Learning resources

- **Databricks Academy**: free self-paced courses (Spark SQL, Delta Lake, MLflow)
- **Hugging Face course**: transformers and text embeddings
- **PyTorch tutorials / fast.ai**: CNNs and transfer learning
- **PyTorch Geometric docs**: GNN basics (stretch goal)
- **SHAP docs**: explainability
- **FastAPI and Streamlit docs**: API and demo

---

## 21. Instructions for Claude Code

When working in this repository:

- **Windows note:** Smart App Control blocks some compiled files and `.exe` launchers. Run tools through the venv interpreter: `.venv\Scripts\python.exe -m pytest`, `-m ruff`, `-m src.ingest.download_bts`. If a package fails with "DLL load failed … contrôle d'application", install a slightly older version (pandas is pinned `<3.1` for this reason).
- **`config.yaml` keys used by the ingest scripts must not be renamed:** `data_dir`, `hubs`, `period.start/end`, `weather_buffer_days`, `radar.product`, `radar.crop_px`, `radar.frames_before_t_pred`, `radar.keep_national_days`, `radar.expected_wld`, `radar.expected_shape`, `t_pred_hours_before_departure`, `iem_sleep_seconds`. Other keys (splits, costs, chain threshold) can be added.

0. **Never run git commands** (no `git add`, `commit`, `push`, `branch`, `checkout`, `pre-commit install`). The author handles version control manually. When a step is done, just list the files that changed.

1. Follow the **Golden rules** (section 9) in all code. Refuse shortcuts that break them.
2. Convert every timestamp to **UTC** before any join; use `airportsdata` time zones.
3. Never use BTS cause columns (`LateAircraftDelay`, `WeatherDelay`, `NASDelay`, `CarrierDelay`, `SecurityDelay`) or the flight's own actual times (`DepTime`, `ArrTime`, `TaxiOut`, `WheelsOff`, ...) as model inputs.
4. All features must be timestamped ≤ `T_pred` (scheduled departure − 2 h).
5. Use **time-based splits only** (train 2024–2025, validate early 2026, test later 2026).
6. Log every training run to **MLflow**.
7. Every join function must log row counts before/after and the match rate.
8. Write **pytest** tests for each new transformation (time, joins, leakage, API).
9. Keep raw data out of git; **never write data inside the repo**. Read `data_dir` and all paths from `config.yaml`.
10. Respect IEM rate limits (≤ 1 request/second, station by station).
11. Prefer small, readable functions with docstrings; explain non-obvious code in comments, because the author must be able to explain every line.
12. When suggesting code, briefly explain **why**, so the author learns, not just copies.

13. Every script takes parameters (dates, airports) so the same code runs on a small sample and on the full scope.
14. Apply the broken-chain rule (section 5) when building rotations.

---

## 22. Related work and our contribution

### Closest existing work
| Work | What it does | What it lacks (vs. this project) |
|---|---|---|
| **Aeolus** (Xu et al., NeurIPS 2025 Datasets & Benchmarks) — https://arxiv.org/abs/2510.26616 · https://github.com/Flnny/Delay-data | BTS 2016–2024 + Meteostat hourly weather; three modalities: tabular, flight chains, flight network graph; temporal splits; tabular, sequence and graph baselines | No radar images; no METAR/TAF text; chains grouped by airline + flight number + date (not tail number); no explicit operational prediction moment; no LLM explanations; no deployed product |
| **Padova student project** — https://github.com/kevinbrugnera/Flight-Delay-Forecasting | Tabular + flight-chain LSTM+MLP fusion on Aeolus | Same gaps as above |
| Many GitHub BTS + METAR projects | Classic ML (Random Forest, LightGBM, XGBoost) on tabular + parsed weather | Not multimodal deep learning; usually no propagation modelling |

### Useful findings to cite
- Aeolus reports modest AUC for tabular models (< 0.63) due to class imbalance: expect difficulty, do not panic.
- Aeolus shows random splits inflate AUC by ~0.057 on average vs. temporal splits: supports Golden rule 3.

### Our contribution (one paragraph for the report)
> Aeolus (NeurIPS 2025) introduced tabular, flight-chain and graph modalities for delay prediction, using hourly weather numbers and chains built from flight numbers. We add two perceptual modalities — radar imagery (CNN) and METAR/TAF weather text (language-model embeddings) — build rotations from real aircraft tail numbers, enforce a strict operational prediction moment using forecasts rather than observations, measure each modality's contribution through ablation, and deliver predictions with fact-checked LLM explanations in a deployed product.

**Never claim** to be "the first" delay-propagation model.

---

## 23. Decisions log

| Date | Decision | Reason |
|---|---|---|
| 2026-10 | Objective changed from ATC readback/hearback error detection to multimodal delay propagation | No public labelled readback/hearback error data; delay data is open, large and linkable |
| 2026-10 | Axis: Operational Efficiency (operational disruption risk) | Fills the empty "Operational efficiency" branch of the mind map; links to Table 7 "Network disruption" |
| 2026-10 | US domestic flights only (BTS) | Only open flight-level delay data with tail numbers |
| 2026-10 | Scope: 10 hubs, train 2025, test 2026 (**supersedes** an early suggestion of calendar year 2024 with 20–30 airports, which was chosen only because ADS-B overlapped then) | Feasible in 6 weeks; tested on recent data; ADS-B dropped so 2024 is no longer needed |
| 2026-10 | `T_pred` = scheduled departure − 2 h | Realistic operational lead time; forces leakage-safe features |
| 2026-10 | ADS-B dropped | Very large, optional, not needed for the core objective |
| 2026-10 | Data stored outside repo/OneDrive; raw files kept | 120 GB storage budget; reproducibility; avoid cloud sync |
| 2026-10 | Tools: Python, DuckDB/Databricks Free Edition, MLflow, PyTorch, FastAPI, Streamlit, Docker, GitHub Actions, Azure for Students | Free; enterprise-relevant (Accenture–Databricks partnership) |
| 2026-10 | Tool policy: core = Databricks, MLflow, pandera, Streamlit + pydeck, Hugging Face Spaces, GitHub Actions, ruff; extras = Power BI, Evidently, Azure, Codecov, demo video; skip Vercel, AWS, Kubernetes | Every tool must solve a real problem; visuals that recruiters see |
| 2026-10 | Inputs include destination weather and radar; tabular encoder explicit; cost-style metric; cloud note + LinkedIn post as deliverables | Arrival conditions affect delays; evaluation and deliverables that recruiters value |
| 2026-10 | Download scripts in `src/ingest/` (BTS, station list, METAR, TAF, radar), resumable, config-driven | Reproducible Bronze layer; same code for sample and full scope |
| 2026-10 | Training period extended to **2024 + 2025** | Two full seasonal cycles: every month (and storm season) seen twice, more rare severe-weather days; pipeline built on 2025 first to limit risk; 1-year vs 2-year comparison reported as an experiment |
| 2026-10 | 2026 data = validation + final test; walk-forward replay and monthly ingestion use it without leakage | Clear separation between learning, tuning and evaluation |
| 2026-10 | Image encoder chosen by a small validation comparison (scratch CNN, ResNet18, EfficientNet-B0, ConvNeXt-Tiny, optional ViT) | Evidence-based choice; radar ≠ natural images |
| 2026-10 | Plan organised in **phases** with exit criteria instead of fixed weeks | Move on as soon as a phase is done; finishing early is allowed |
| 2026-10 | Storage 120 GB → radar `n0q` kept with crop-and-delete; crops stored compactly | High quality without filling the disk |
| 2026-10 | Deep training on Kaggle GPU; Databricks for joins/Delta/MLflow; DuckDB as local fallback | Databricks Free Edition has no GPU and daily quotas |
| 2026-10 | PostgreSQL not needed (Could): the demo reads precomputed predictions from Parquet/DuckDB | Less setup; focus on the model |
| 2026-10 | No separate "sample phase": exploration uses one month; code is parameterised by dates and run on everything once correct | Saves time under the 6-week deadline |
| 2026-10 | Git handled manually by the author; AI assistants never run git commands | Full control over history |
| 2026-10 | Rotation chains built per tail over continuous UTC time; a ground time above the data-chosen threshold (`chain_break_ground_time_hours`, set in Phase 1 step 1.2) starts a new chain | Overnight rest/maintenance absorbs delays; red-eyes cross midnight |
| 2026-10 | Chain break threshold chosen from the ground-time distribution; overnight stops and maintenance handled as chain breaks | Delay propagation mostly happens within an operating day |
| 2026-10 | PC Parquet = master copy; develop joins in DuckDB; run the real Bronze/Silver/Gold pipeline + MLflow on Databricks; export Gold back | Real Databricks work for the CV, no data split, quota-proof |
| 2026-10 | Renamed airports handled by `airport_aliases` in `config.yaml` (first case: `DJT` → `PBI`; used by `build_station_list` and `radar_needed_hours`); Silver keys airports on BTS `OriginAirportID` / `DestAirportID`; IEM weather ids that differ from the code go in `weather_station_ids` (PBI: METAR `DJT`, TAF `KPBI` + `KDJT` as two raw files, merged in Silver) | One physical airport = one station and one category for the model; BTS codes can change mid-period (PBI → DJT, Aug 2026), and IEM handles the rename differently for METAR (history moved) and TAF (archive split) |
| 2026-11 | Radar crop size **256 → 384 px** (≈ 213 km N–S, ≈ 178 km E–W at ATL); old 256 px crops deleted, never mixed (`download_radar` checks the size of existing `.npz` files) | A storm moving 40–50 km/h covers ~100 km in the 2 h between `T_pred` and departure; 256 px saw only ~±70 km. Same download time (same national images); storage ~34 GB of crops instead of ~15 GB |
| 2026-10 | Repo private at start, public after cleanup (end of Phase 1–2) | Clean first impression; unlimited Actions minutes on public repos |
