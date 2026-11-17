"""I/O driver for the GTA VI Twitch CCU forecaster.

Reads every <id>.in in /input (or $MAKEFUN_INPUT_DIR) and writes one
<id>.out per case to /output (or $MAKEFUN_OUTPUT_DIR).  The model lives in
forecast.py; this file only does the file handling.  Standard library
only, deterministic, no network.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forecast


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(in_path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast.forecast_case(case)
        except Exception:
            # Never fail the whole run on one bad case: emit the fallback
            # shape at a print of 1 viewer so the output stays valid.
            result = {"hours": [
                {"hour": h,
                 "q": forecast._quantiles(
                     1.0, forecast.FALLBACK_SHAPE[h],
                     forecast.FALLBACK_SPREAD, forecast.DEFAULT_LEVELS)}
                for h in forecast.HOURS]}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
