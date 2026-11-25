#!/usr/bin/env python3
"""Run ``rev_explain`` on every arm-A ProtoHedge fit of a study (both layers).

    python scripts/revision/rev_explain_all.py --spec paper/revision_2026_11/phase3/final_spec.json --split val
    python scripts/revision/rev_explain_all.py --spec ... --split test --allow-test   # only after the final evaluation

With ``--split test`` it refuses to run unless the final evaluation has already
opened the test episodes (``final/TEST_OPENED.json``).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev_core as rc  # noqa: E402,F401
import phase3_fit  # noqa: E402
import rev_explain  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--allow-test", dest="allow_test", action="store_true")
    args = p.parse_args()
    spec = json.load(open(args.spec))
    if args.split == "test":
        if not (phase3_fit.ROOT / spec["study"] / "final" / "TEST_OPENED.json").exists() or not args.allow_test:
            raise SystemExit("The test episodes are opened by phase3_final.py first; then pass --allow-test.")
    arms = [a for a in spec["arms"] if a["name"] in ("ph", "phd")]
    n = 0
    for arm in arms:
        for liability in spec["liabilities"]:
            for ticker in spec["tickers"]:
                for seed in arm["seeds"]:
                    ns = argparse.Namespace(study=spec["study"], ticker=ticker, liability=liability, mode=arm["mode"],
                                            objective=arm["objective"], state=arm["state"], seed=seed,
                                            family=arm["family"], split=args.split, allow_test=args.allow_test,
                                            step=10, top=3)
                    with contextlib.redirect_stdout(io.StringIO()):
                        rev_explain.explain(ns)
                    n += 1
            print(f"{time.strftime('%T')} {arm['name']} {liability}: {n} fits explained", flush=True)


if __name__ == "__main__":
    main()
