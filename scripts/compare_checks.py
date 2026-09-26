"""Compare two `check_avatar.py --json` reports, to see what a rule change did.

    python scripts/compare_checks.py before.json after.json [--worse-than 0.5]

Rules here are tuned by trying them on a character and looking; the risk is that a rule that
helps one model quietly breaks another. This prints every pose's stretch and squash side by
side and flags what got worse, so a change can be shown to have left the other characters
alone. Keep a report per character (`make_avatar.py` writes one as `<name>_check.json`) and
compare after every change.

Exit code 1 when something got worse by more than the margin, so it can gate a change.
"""

import argparse
import json
import sys
from pathlib import Path

# Changes smaller than these are noise from the voxel remesh, not a rule doing something.
STRETCH_NOISE = 0.5  # cm
SQUASH_NOISE = 0.03  # share of the edge's rest length


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--worse-than", type=float, default=STRETCH_NOISE,
                        help="how many cm of extra stretch counts as worse (default %(default)s)")
    args = parser.parse_args()
    before, after = (json.loads(p.read_text()) for p in (args.before, args.after))

    print(f"{'pose':26s} {'stretch cm':>20s}   {'squash':>16s}")
    for pose in sorted(set(before["poses"]) | set(after["poses"])):
        old, new = before["poses"].get(pose, {}), after["poses"].get(pose, {})
        print(f"{pose:26s} {_pair(old.get('stretch_cm'), new.get('stretch_cm')):>20s}   "
              f"{_pair(old.get('squash'), new.get('squash')):>16s}")
    print(f"{'unweighted vertices':26s} {_pair(before['unweighted'], after['unweighted']):>20s}")
    worse, _ = changes(before, after, args.worse_than)
    print()
    if worse:
        print("WORSE")
        for line in worse:
            print(f"  {line}")
        sys.exit(1)
    print("nothing got worse")


def changes(before, after, worse_than=STRETCH_NOISE):
    """(worse, better): lines naming each pose whose stretch or squash moved past the noise."""
    worse, better = [], []
    for pose in sorted(set(before["poses"]) | set(after["poses"])):
        old, new = before["poses"].get(pose, {}), after["poses"].get(pose, {})
        if _grew(old.get("stretch_cm"), new.get("stretch_cm"), worse_than):
            worse.append(f"{pose}: stretch {old['stretch_cm']} -> {new['stretch_cm']} cm")
        elif _grew(new.get("stretch_cm"), old.get("stretch_cm"), worse_than):
            better.append(f"{pose}: stretch {old['stretch_cm']} -> {new['stretch_cm']} cm")
        if _grew(old.get("squash"), new.get("squash"), SQUASH_NOISE, smaller_is_worse=True):
            worse.append(f"{pose}: squash {old['squash']} -> {new['squash']}")
        elif _grew(new.get("squash"), old.get("squash"), SQUASH_NOISE, smaller_is_worse=True):
            better.append(f"{pose}: squash {old['squash']} -> {new['squash']}")
    if after.get("unweighted", 0) > before.get("unweighted", 0):
        worse.append(f"unweighted vertices {before['unweighted']} -> {after['unweighted']}")
    return worse, better


def _pair(old, new):
    if old is None and new is None:
        return "-"
    return f"{_number(old)} -> {_number(new)}"


def _number(value):
    return "-" if value is None else f"{value:g}"


def _grew(old, new, margin, smaller_is_worse=False):
    if old is None or new is None:
        return False
    return (old - new > margin) if smaller_is_worse else (new - old > margin)


if __name__ == "__main__":
    main()
