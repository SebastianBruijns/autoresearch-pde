"""One command: data in -> equation, key steps, confidence, and what to do next.

    python -m eqdisc.discover path/to/data.(csv|mat|npz|h5|json) | datasets/<dir>
        [--branches 3] [--no-adversary] [--human] [--context "closed population; mass-action"]
        [--max-tools 20] [--effort high]
"""
import argparse

from .orchestrate import discover


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--branches", type=int, default=3)
    p.add_argument("--no-adversary", action="store_true")
    p.add_argument("--human", action="store_true")
    p.add_argument("--context")
    p.add_argument("--max-tools", type=int, default=20)
    p.add_argument("--effort", default="high")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--out")
    a = p.parse_args()
    from pathlib import Path
    if not Path(a.path).exists():
        ex = sorted(str(d) for d in Path("datasets").glob("*") if (d / "meta.json").exists())[:12]
        p.error(f"'{a.path}' does not exist. Give a data file (csv/mat/npz/h5/json) or a dataset folder, e.g.\n  "
                + "\n  ".join(ex) + "\n  examples/data/KS_data.mat")
    human = (lambda q: input(f"\n{'=' * 70}\n{q}\n> ")) if a.human else None
    discover(a.path, a.branches, not a.no_adversary, human, a.context, a.model, a.effort, a.max_tools, a.out)


if __name__ == "__main__":
    main()
