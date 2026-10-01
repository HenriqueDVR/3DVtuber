"""Run the whole suite on every Blender found, one after another, and print a table.

    python tests/run_matrix.py [--blender PATH ...] [-- <test file pattern>]

Plain Python. Finds the installed Blenders (app/engine.py installed_blenders) and the portable ones
unzipped under tools/blender/ (each with a `portable` folder, so their add-ons stay apart from the
installed Blender's). Runs strictly one at a time: two Blenders at once hung the machine.
A series that passes goes into engine.TESTED.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import engine  # noqa: E402

LOGS = REPO / "out" / "matrix"


def blenders():
    found = list(engine.installed_blenders()) + sorted((REPO / "tools" / "blender").glob("*/blender.exe"))
    return [p for p in found if (engine._folder_version(p) or (0,)) >= engine.MIN_BLENDER]


def run(blender, pattern):
    blender = Path(blender).resolve()  # Windows won't start a relative path with forward slashes
    version = subprocess.run([str(blender), "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / (re.sub(r"[^\w.]+", "_", version).strip("_") + ".log")
    command = [str(blender), "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(REPO / "tests" / "run_tests.py")]
    if pattern:
        command += ["--", pattern]
    started = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as handle:
        code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, cwd=REPO).returncode
    text = log.read_text(encoding="utf-8", errors="replace")
    ran = re.search(r"^Ran (\d+) tests", text, re.MULTILINE)
    verdict = re.search(r"^(OK.*|FAILED.*)$", text, re.MULTILINE)
    failing = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+ \([^)]+\))", text, re.MULTILINE)))
    return {"blender": version, "path": str(blender), "exit": code, "tests": int(ran.group(1)) if ran else None,
            "verdict": verdict.group(1) if verdict else "no result (see log)", "failing": failing,
            "minutes": round((time.time() - started) / 60, 1), "log": str(log)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", action="append", help="run only these (repeatable)")
    args, rest = parser.parse_known_args()
    pattern = rest[rest.index("--") + 1] if "--" in rest and rest.index("--") + 1 < len(rest) else None
    results = []
    for blender in args.blender or blenders():
        print(f"running the suite on {blender} ...", flush=True)
        results.append(run(blender, pattern))
        print(json.dumps(results[-1], indent=2), flush=True)
    print("\n{:<40} {:>6} {:>6}  {}".format("Blender", "tests", "min", "verdict"))
    for r in results:
        print("{:<40} {:>6} {:>6}  {}".format(r["blender"], r["tests"] or "-", r["minutes"], r["verdict"]))
        for name in r["failing"]:
            print(f"    {name}")
    return 0 if all(r["exit"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
