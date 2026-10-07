---
name: vrdet-research
description: Continue the autonomous VRDet research loop (new commercially-safe OBB detector, benchmarked like YOLO-OBB on DOTA) - analyse finished Colab jobs, decide and launch the next experiment via tools/job.py, babysit it, record results for the next agent.
---

# VRDet research (Claude Code adapter)

The canonical, agent-neutral instructions live in
`.agents/skills/vrdet-research/SKILL.md`. Read that file now and follow it exactly.

Claude Code specifics:
- Run `python tools/job.py watch` with `run_in_background: true`. When it exits you are re-invoked: harvest the results and continue the loop without waiting for the user.
- On Windows, use the Bash tool with `export PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1`. The interpreter is `/c/Users/HP/AppData/Local/Programs/Python/Python312/python.exe`; plain `python` is not on PATH in Git Bash.
