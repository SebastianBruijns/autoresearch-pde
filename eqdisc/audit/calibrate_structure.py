"""Calibrate / evaluate the structural priors and checks on DEV datasets (no LLM calls).

    python -m eqdisc.audit.calibrate_structure [--glob 'datasets/*_s[012]'] [--out runs/structure_calibration]

Per dataset (1-D periodic PDEs only), three arms, scored with evaluate(reveal=True):
  base     weak_sindy with default arguments
  priors   weak_sindy with the programme implied by audit/priors.py
  polished priors + structure.polish (one-term-at-a-time repairs)
  checks   base + structure.polish (the checks alone, without priors)
plus the false-alarm check: structure.audit on the TRUE equation must fire no critical finding.
Uses hidden truth only to MEASURE; never run it on reporting seeds or eqdisc.blind held-out systems.
"""
import argparse
import glob
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ARMS = ("base", "priors", "polished", "checks")


def run_one(d):
    from ..evaluate import evaluate, load
    from ..weakform import weak_sindy
    from . import priors, structure
    meta, data = load(d)
    if not priors.applicable(meta, data):
        return None
    truth = json.loads((Path(d) / "hidden" / "truth.json").read_text())
    row = {"dataset": Path(d).name, "system": truth["system"], "noise": truth["noise"]}
    t0 = time.time()
    card = priors.card(meta, data)
    arms = {"base": weak_sindy(meta, data)["rhs"],
            "priors": weak_sindy(meta, data, **card["programme"]["args"])["rhs"]}
    arms["polished"] = structure.polish(meta, data, arms["priors"])["rhs"]
    arms["checks"] = structure.polish(meta, data, arms["base"])["rhs"]
    for k, rhs in arms.items():
        e = evaluate(d, rhs, reveal=True)
        row[k] = {"rhs": rhs, "score": round(e["score"], 3), "f1": round(e["f1"], 3), "exact": e["exact_structure"]}
    tf = structure.audit(meta, data, truth["rhs"])
    row["truth_false_alarms"] = [f["id"] for f in tf if f["fired"] and f["severity"] == "critical"]
    row["priors_excluded"] = card["programme"]["args"].get("exclude_terms", [])
    row["seconds"] = round(time.time() - t0, 1)
    return row


def summarise(rows):
    out = {"n": len(rows)}
    for k in ARMS:
        out[k] = {"exact_rate": round(sum(r[k]["exact"] for r in rows) / len(rows), 3),
                  "mean_f1": round(sum(r[k]["f1"] for r in rows) / len(rows), 3),
                  "mean_score": round(sum(r[k]["score"] for r in rows) / len(rows), 3)}
    out["truth_false_alarm_rate"] = round(sum(bool(r["truth_false_alarms"]) for r in rows) / len(rows), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="datasets/*_s[012]")
    p.add_argument("--out", default="runs/structure_calibration")
    a = p.parse_args()
    rows = []
    for d in sorted(glob.glob(a.glob)):
        if "blind" in d or not (Path(d) / "hidden" / "truth.json").exists():
            continue
        r = run_one(d)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['dataset']:40s} " + " ".join(f"{k}={r[k]['f1']:.2f}{'*' if r[k]['exact'] else ' '}"
                                                 for k in ARMS)
              + f" false_alarms={r['truth_false_alarms']} {r['seconds']}s", flush=True)
    s = summarise(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
