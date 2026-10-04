"""Append-only evidence ledger: one JSON line per finding, repair attempted, re-audit outcome and tournament outcome.

    led = Ledger(run_dir, dataset_dir, config={...})
    led.append("finding", **f)            # every line carries time, dataset_hash (sha256 of data.npz), config_hash

Written to `<run_dir>/ledger.jsonl`. Lines are never rewritten; a re-scored run (reassess) appends new lines.
"""
import hashlib
import json
import time
from pathlib import Path


def file_hash(path):
    """sha256 of a file (None if missing)."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_hash(dataset_dir):
    return file_hash(Path(dataset_dir) / "data.npz") if dataset_dir else None


def config_hash(config):
    return hashlib.sha256(json.dumps(config or {}, sort_keys=True, default=str).encode()).hexdigest()


class Ledger:
    def __init__(self, run_dir, dataset_dir=None, config=None):
        self.path = Path(run_dir) / "ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_hash = dataset_hash(dataset_dir)
        self.config_hash = config_hash(config)

    def set_dataset(self, dataset_dir):
        self.dataset_hash = dataset_hash(dataset_dir)

    def append(self, kind, **fields):
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, "dataset_hash": self.dataset_hash,
               "config_hash": self.config_hash, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def findings(self, findings, phase):
        for f in findings or []:
            self.append("finding", phase=phase, **{k: f.get(k) for k in (
                "id", "stage", "statistic", "threshold", "fired", "severity", "response", "fix", "scope", "message",
                "resolved")})


def read(run_dir):
    p = Path(run_dir) / "ledger.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []
