import pathlib
import subprocess
import sys


def test_simulated_run():
    root = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "snap_delta_bot.py"), "--simulate"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Entered" in result.stdout
