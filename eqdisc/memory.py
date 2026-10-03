"""Cross-session experience memory (lessons learned), retrieved into the agent's system prompt.

After each session a short reflection (written by the model) is stored with a fingerprint of the
dataset (kind, #vars, noise, tags from diagnose). New sessions retrieve the most similar lessons.
Lessons are de-duplicated by text similarity so the memory stays diverse (STRIDE semantic memory).
"""
import json
import re
import time
from pathlib import Path

MEMORY = Path(__file__).resolve().parent.parent / "memory" / "lessons.jsonl"


def fingerprint(meta, diag=None):
    noise = None
    if diag and "noise_rel_estimate" in diag:
        noise = max(diag["noise_rel_estimate"].values())
    return {"kind": meta["kind"], "n_vars": len(meta["variables"]), "noise": noise,
            "spatial_dims": len(meta.get("spatial_dims", ["x"])) if meta["kind"] == "pde" else 0,
            "boundary": meta.get("boundary")}


def _words(s):
    return set(re.findall(r"[a-z_]{3,}", s.lower()))


def load_all(path=MEMORY):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def add_lessons(lessons, fp, outcome, path=MEMORY):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_all(path)
    kept = 0
    with open(path, "a") as f:
        for text in lessons:
            w = _words(text)
            if any(len(w & _words(e["lesson"])) / (len(w | _words(e["lesson"])) + 1e-9) > 0.6 for e in existing):
                continue                                   # near-duplicate: keep memory diverse
            rec = {"lesson": text.strip(), "fingerprint": fp, "outcome": outcome, "time": time.time()}
            f.write(json.dumps(rec) + "\n")
            existing.append(rec)
            kept += 1
    return kept


def retrieve(meta, diag=None, k=8, path=MEMORY):
    fp = fingerprint(meta, diag)
    recs = load_all(path)

    def sim(r):
        g = r["fingerprint"]
        s = 2.0 * (g["kind"] == fp["kind"]) + 0.5 * (g["n_vars"] == fp["n_vars"])
        s += 0.5 * (g.get("spatial_dims") == fp.get("spatial_dims")) + 0.3 * (g.get("boundary") == fp.get("boundary"))
        if fp["noise"] is not None and g.get("noise") is not None:
            s += 1.0 / (1.0 + abs(g["noise"] - fp["noise"]) * 20)
        s += 0.3 * (r.get("outcome", {}).get("good", False))
        return s
    recs.sort(key=sim, reverse=True)
    return [r["lesson"] for r in recs[:k]]


REFLECT_PROMPT = """You just finished an equation-discovery session. Summary of the session (tools called, key
results, final model{outcome_hint}):

{summary}

Write 2-4 short, GENERAL lessons for future sessions on other datasets: concrete decision rules (what to try
first given what diagnostics, which settings worked or failed and why, when to switch tools). No
dataset-specific equations. Return ONLY a JSON list of strings."""
