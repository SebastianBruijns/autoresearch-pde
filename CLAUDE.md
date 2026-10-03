# eqdisc

Autoresearch for governing equations. When a user wants equations or dynamics discovered from data, use the
`discover-equations` skill (`.claude/skills/discover-equations/SKILL.md`). The package lives in `eqdisc/`. The agent's
own domain guidance is in `eqdisc/skills/*.md`, and its strategy in `eqdisc/playbook.md`; both ship with the package.

Development rules:
- Never tune thresholds or heuristics on the held-out blinded benchmark (`eqdisc.blind`). Report development and
  held-out systems separately.
- Domain skills give method guidance, never the closed-form answers of benchmark systems.
- Smoke test without API calls: `python -m eqdisc.tests.smoke`.
