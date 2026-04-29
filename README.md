# HULHE Bot

This project implements a heads-up limit Texas hold'em abstraction and training
pipeline with a stdlib-first runtime and optional LiteEFG/OpenSpiel integration.

Main entrypoints:

- `hulhe-bot build_abstract_game`
- `hulhe-bot train_blueprint`
- `hulhe-bot fine_tune`
- `hulhe-bot evaluate`
- `hulhe-bot solver_check`
- `hulhe-bot play_ui`

Lightweight local UI:

- Start a browser UI to play against a loaded policy artifact:
	- `python -m hulhe_bot.cli play_ui --config configs/tag_heavy_045_cfr_subgame.json --policy artifacts/tag_heavy_045_tuned.json`
	- Open `http://127.0.0.1:8765`

Optional native exact-river solver:

- The exact river resolver can use an optional C++ extension and falls back to
  the Python reference implementation if the extension is unavailable.
- Build it in-place with:
	- `./poker-rl/bin/pip install pybind11`
	- `./poker-rl/bin/python build_native_solver.py build_ext --inplace`
