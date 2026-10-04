# eqdisc

Autoresearch for governing equations. When a user wants equations or dynamics discovered from data, use the
`discover-equations` skill (`.claude/skills/discover-equations/SKILL.md`). The package lives in `eqdisc/`. The agent's
strategy is in `eqdisc/playbook.md` (domain-agnostic method guidance only); it ships with the package.

Development rules:
- Never tune thresholds or heuristics on the held-out blinded benchmark (`eqdisc.blind`). Report development and
  held-out systems separately.
- No domain knowledge goes to the agent: no domain skills, no context naming the physical system, no dataset names
  that reveal it. The agent gets data only, and must recover the equations from it.
- Smoke test without API calls: `python -m eqdisc.tests.smoke`.
