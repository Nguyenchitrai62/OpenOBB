---
name: vrdet-research
description: Continue the autonomous VRDet research loop (new commercially-safe OBB detector architecture, benchmarked like YOLO-OBB on DOTA) in this repo. Reads the current state, analyses finished Colab jobs, decides and launches the next experiment on Google Colab via tools/job.py, babysits it, and records everything so any other coding agent can pick up where this one stopped.
---

# VRDet research loop

Run this when the user invokes `$vrdet-research` / `/vrdet-research`, or asks to
"tiếp tục nghiên cứu" / "continue the research" in this repo. The user is usually
**not online**: work autonomously, decide by yourself, and never block waiting for
a click. Things that truly need the human go into the "Cần user" list in
`AGENTS.md` §8. Then keep working on whatever does not depend on them.

## 0. Load context (always, in this order)

1. `AGENTS.md`: goal, hard constraints, current status (§8), log (§9).
2. `docs/RESEARCH_PLAN.md`: benchmark protocol, baselines to beat, hypotheses H1–H7, roadmap E0–E6.
3. `docs/AUTONOMY.md`: autonomy rules and **hard limits** (budget, foreign sessions, secrets).
4. `docs/COLAB.md`: Colab CLI usage. On Windows: the two local patches, plus `PYTHONIOENCODING=utf-8` and `MSYS_NO_PATHCONV=1`.
5. `research/HYPOTHESES.md` and the tail of `research/decisions.log`.

## 1. Preflight

```bash
python tools/job.py status          # jobs + CU balance (Windows: py -3.12 tools/job.py ...)
colab whoami                         # Colab auth still valid?
colab sessions                       # '[?]' rows belong to the user: never touch them
```

- `colab whoami` fails → you cannot train. Add a "Cần user: đăng nhập lại Colab (docs/COLAB.md §3)" item, then do offline work only (code, unit tests, analysis, docs).
- `colab` missing → `uv tool install google-colab-cli`. On Windows, re-apply the patches described in `docs/COLAB.md` §1.

## 2. One iteration of the loop

1. **Harvest.** For every job whose state is not `running` and that `research/decisions.log` does not mention yet:
   - read `runs/<id>/metrics.jsonl`, `train.log`, `setup.log`;
   - write the verdict (confirmed / rejected / inconclusive, with numbers) into `research/HYPOTHESES.md`;
   - append one line to `research/decisions.log`: `[time] <id> status=<s> result=<key numbers> cu=<spent> next=<decision>`.
   - Failed jobs: diagnose from the logs, fix the code, relaunch. Do not relaunch an unchanged spec more than once.
2. **Decide.** Pick the next experiment from the roadmap (`docs/RESEARCH_PLAN.md` §5), in order, unless evidence says otherwise.
   - Before launching, write the hypothesis, the single variable being changed, and the pass/fail criterion into `research/HYPOTHESES.md`.
   - Always keep a same-conditions baseline. Prefer the cheapest experiment that can falsify the hypothesis (DIOR-R or a DOTA subset with a short schedule) before a full DOTA run.
3. **Implement.** Code lives in `vrdet/` (model, losses, data, eval) and `colab/` (VM-side scripts, `colab/data/*.sh` dataset fetchers).
   - Training scripts must write everything to `--out`, auto-resume from `{out}/last.pt`, and append one JSON line per epoch to `{out}/metrics.jsonl`.
   - Setup steps must be idempotent: they rerun on every fresh VM.
   - Test locally on CPU with a tiny config before spending GPU (`python -m pytest -q` if tests exist).
4. **Launch.** Write `research/jobs/<id>.json` (format: docstring of `tools/job.py`; set `max_hours`), then run `python tools/job.py launch <id>`.
   - Job ids look like `e2-h1-dior-s`.
   - GPU choice: **G4** (RTX PRO 6000 Blackwell, 94 GB, ~8.9 CU/h) is the default for every real experiment, as chosen by the user. T4 (~1.1 CU/h) is for smoke tests only. A100 is the fallback if G4 cannot be allocated.
5. **Babysit.** Run `python tools/job.py watch`. It polls every 10 min, syncs checkpoints to `runs/<id>/`, relaunches on a new VM and resumes if the VM dies, and stops VMs when jobs end or the balance nears the 40 CU reserve. It exits when a job ends (or after 8 h).
   - Claude Code: run it with `run_in_background: true`; you are re-invoked when it exits.
   - Agents without background notification: run it in the foreground.
   - When it exits, go back to step 1.
6. **Record.** After each iteration:
   - update `AGENTS.md` §8 (checkboxes, current job, "Cần user") and add a line to §9;
   - commit: `git add -A && git commit -m "<what changed / what was learned>"`. Never commit `.env`, datasets or weights; `.gitignore` covers them.

## 3. Stop conditions

Stop the loop, after recording state, when any of these holds:
- the user says stop;
- the balance is below the reserve;
- Colab auth is broken;
- the goal in `RESEARCH_PLAN.md` §4.3 is reached and documented.

If you are about to run out of context or quota, first make sure `AGENTS.md` §8–9 describes exactly what is running and what comes next. Running jobs keep going on Colab, and the next agent continues with `python tools/job.py watch`.

## 4. Rules that are easy to forget

- Never use Ultralytics code inside `vrdet/` (AGPL). It may only be run as a baseline.
- Use the standard DOTA rotated-mAP evaluator. Never claim a win without numbers measured under identical conditions.
- Before ending a turn, run `colab sessions` and make sure no VM of yours is idle.
- Never type passwords or OAuth codes for the user.
