---
name: discover-equations
description: Discover governing ODE/PDE equations from measured or simulated data with the eqdisc autoresearch framework. It returns the equation, the key steps taken, confidence (UQ) and a recommendation (confident, or where to collect more data). Use it when the user gives a data file or folder (csv, mat, npz, h5, json, time series or spatio-temporal fields) and wants the underlying dynamics, governing equations, a symbolic model, SINDy/PySR-style discovery, or advice on what data to collect next.
---

# Discover equations from data (eqdisc)

This repository is the Python package `eqdisc`. One command runs the full pipeline:
ingest → intuition pre-analysis → parallel agent branches → tournament → adversarial red team → assessment → HTML report.

## 1. Setup (once)
- Install the package: `pip install -e .` (add `".[pysr]"` for symbolic regression; PySR installs Julia on first use).
- Credentials for the Claude API: `ant auth login` (Anthropic CLI), or `ANTHROPIC_API_KEY=...` in the environment or
  in a `.env` file at the repo root. Check with `ant auth status`. If neither is set, ask the user to set one up.

## 2. Before running
- Run data-only by default: do not ask for, or add, domain context. Pass `--context` only if the user volunteers
  it, and say in the result that context was used (it can steer the answer toward what the user already believes).
- Cost and time: about $0.3–0.5 per agent branch, typically $1–3 and 2–6 minutes for 3 branches plus the
  adversary (more for large or hard problems). Tell the user before running many datasets or more than 3 branches.

## 3. Run (in the background; it prints the verdict and report path when done)
```bash
eqdisc-discover <PATH> --branches 3 --context "<domain knowledge>"
```
- `<PATH>`: a raw data file (it is ingested automatically; read the printed data-card warnings) or a dataset folder.
- `--no-adversary --branches 2` for a quick, cheaper pass. `--human` for interactive review in a terminal.
- Example: `eqdisc-discover examples/data/KS_data.mat`

## 4. Report back (read `runs/discover_*/discovery.json`)
Lead with the verdict and the equation, then:
1. `verdict.status` (CONFIDENT / CONFIDENT IN PREDICTIONS / COLLECT MORE DATA / INCONCLUSIVE), with its headline
   and recommendation.
2. `final_model`, in readable math.
3. Key steps: `story.key_steps` and `insights` (e.g. "spatial mean conserved → flux form").
4. Confidence reasons, and per-term intervals from `assessment.terms`.
5. Next experiments: the top `assessment.experiments.ranked` entries, with what each would pin down.
6. `assessment.questions_for_human`: ask them; the answers can be fed back with `--context`.
7. Link `report.html`. If `benchmark` is present (synthetic data with known truth), say that it is a benchmark score.
Be honest about weak results: if branches disagree or the verdict is not CONFIDENT, say so plainly.

## 5. Follow-ups
- New data from a recommended experiment: re-run on the combined data, or use `eqdisc-agent <dataset> --human` and
  reply `data: <path>` at the checkpoint.
- A single cheaper agent session: `eqdisc-agent <dataset> --max-tools 15`.
- Visual walkthroughs: `notebooks/01..05`.
