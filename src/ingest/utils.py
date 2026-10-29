"""Shared helpers for the download scripts: config loading, folders, polite HTTP."""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "multimodal-flight-delay-propagation (student research project)"


def load_config() -> dict:
    """Read config.yaml at the repo root."""
    with open(REPO_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_dir(cfg: dict, source: str) -> Path:
    """Return <data_dir>/raw/<source>, creating it if needed."""
    path = Path(cfg["data_dir"]) / "raw" / source
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_date(value) -> date:
    """Accept a date object or a 'YYYY-MM-DD' string."""
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def period(cfg: dict, buffer_days: int = 0) -> tuple[date, date]:
    """Start/end dates of the study period, optionally widened at the start."""
    start = to_date(cfg["period"]["start"]) - timedelta(days=buffer_days)
    end = min(to_date(cfg["period"]["end"]), date.today())
    return start, end


def http_get(url: str, params: dict | None = None, stream: bool = False,
             retries: int = 4, timeout: int = 120) -> requests.Response | None:
    """GET with retries and back-off. Returns None on 404 (file does not exist)."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, stream=stream,
                             timeout=timeout, headers=headers)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.RequestException, requests.HTTPError) as exc:
            wait = 5 * attempt
            print(f"   ! attempt {attempt}/{retries} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}")


def save_atomic(path: Path, content: bytes) -> None:
    """Write to a temporary file then rename, so a crash never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def update_no_data(path: Path, attempted: set[str], failures: dict[str, str]) -> None:
    """Rewrite the no-data list: this run's failures + earlier entries not retried this run.

    A station retried this run is dropped from the old list (it either succeeded, or it is
    in `failures` again), so nothing is lost silently and fixed stations disappear.
    """
    old = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old[line.split()[0]] = line
    kept = {code: line for code, line in old.items() if code not in attempted}
    kept.update({code: f"{code}  tried: {tried}" for code, tried in failures.items()})
    path.write_text("".join(f"{kept[c]}\n" for c in sorted(kept)), encoding="utf-8")
