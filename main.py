"""gta6-twitch-ccu-forecast-01 entrypoint: GTA VI Twitch CCU forecaster.

Reads every case file <id>.in in the input directory and writes one
forecast <id>.out per case to the output directory. Each forecast gives,
for hours 2..24 of the release day, five quantiles of that hour's mean
Twitch viewers in the Grand Theft Auto VI category.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts
it back first, letting `import forecast` find the model shipped beside
this file; -s drops user site-packages.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The forecasting model lives in forecast.py; this file is only the I/O
driver: read cases in name order (so two runs write the same bytes),
forecast each one, write the compact JSON output.
"""

import json
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# forecast` finds the model shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import forecast  # noqa: E402  (after the sys.path line above, on purpose)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast.forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
