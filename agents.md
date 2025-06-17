# Snap‑Delta Bot – Contributor Guide

## Quick Start

1. `pip install -r requirements.txt`
2. Copy `.env.example` → `.env` and fill with your Alpaca paper‑trading keys (and optional `POLYGON_API_KEY`).
3. Run a smoke test: `python snap_delta_bot.py --simulate`  # emits dummy ticks.

## Validation Rules

- **Unit tests**: All PyTest tests must pass – run `pytest -q`.
- **Lint**: `ruff check .` should return no errors.
- **Formatting**: Use `black --check .` (line length ≤ 100).

## Agent Tasks

- **Ask‑mode** – architecture Q&A, performance analysis, brainstorming.
- **Code‑mode** – code modifications, refactors, test creation.
  - After code changes, run `pytest -q` and paste the output in the PR description.

## Project Layout

```
.
├── snap_delta_bot.py         # Core trading engine
├── requirements.txt          # Python deps
├── Dockerfile                # Container build file
├── docker-compose.yml        # Local orchestration (optional)
├── tests/
│   └── test_bot.py           # Smoke test
└── .env.example              # Env‑var template
```

## Smoke‑Test Definition

The smoke test launches the bot in simulation mode for **3–5 ticks** and asserts:

- Trade engine initializes without exceptions.
- At least one signal is generated.
- Process exits with status 0.

## Style Guide (code review gate)

- Full type hints (`from __future__ import annotations`).
- Prefer `asyncio`/`aiohttp` libs over blocking I/O.
- Keep functions ≤ 40 LOC; extract helpers otherwise.

## Required Environment Variables

| Variable              | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| `APCA_API_KEY_ID`     | Alpaca key                                          |
| `APCA_API_SECRET_KEY` | Alpaca secret                                       |
| `APCA_API_BASE_URL`   | Set to `https://paper-api.alpaca.markets` for paper |
| `POLYGON_API_KEY`     | (opt) Option chain snapshots                        |

Happy shipping! 🚀

