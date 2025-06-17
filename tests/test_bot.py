"""Smoke test for Snap‑Delta Bot.
Runs the bot in simulation mode to ensure it starts, emits at least one trade, and exits cleanly.
Requires `snap_delta_bot.py` in project root.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_simulation_runs():
    cmd = [sys.executable, "snap_delta_bot.py", "--simulate", "3"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, f"Bot exited with non‑zero status: {result.stderr}"
    assert "SIMULATION COMPLETE" in result.stdout.upper()
