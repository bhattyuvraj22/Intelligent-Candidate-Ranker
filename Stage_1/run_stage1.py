"""
Stage 1 Runner — Hard Filter + Honeypot Flag Pipeline
Streams candidates.jsonl line by line (no full memory load).

Outputs:
  stage1_passed.jsonl   — survivors with _honeypot_flags + _location_multiplier attached
  stage1_rejected.jsonl — rejected candidates with _stage1_rejected reason tag

Usage:
  python3 run_stage1.py --input /path/to/candidates.jsonl --out_dir ./output
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from stage1_rules import apply_all_filters


def run(input_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    passed_path = out_dir / "stage1_passed.jsonl"
    rejected_path = out_dir / "stage1_rejected.jsonl"

    total = passed = 0
    filter_counts = Counter()
    honeypot_counts = Counter()
    location_penalized = 0
    tier2_rescued = 0
    start = time.time()

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(passed_path, "w", encoding="utf-8") as fpass,
        open(rejected_path, "w", encoding="utf-8") as frej,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {total}: {e}", file=sys.stderr)
                continue

            ok, reason = apply_all_filters(candidate)

            if ok:
                fpass.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                passed += 1
                # Track soft signals for summary
                if candidate.get("_location_multiplier", 1.0) < 1.0:
                    location_penalized += 1
                if candidate.get("_tier2_rescue"):
                    tier2_rescued += 1
                for flag in candidate.get("_honeypot_flags", []):
                    honeypot_counts[flag] += 1
            else:
                candidate["_stage1_rejected"] = reason
                frej.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                tag = reason.split("|")[0].strip() if reason else "unknown"
                filter_counts[tag] += 1

            if total % 10_000 == 0:
                elapsed = time.time() - start
                print(f"  Processed {total:,} | passed: {passed:,} | {elapsed:.1f}s")

    elapsed = time.time() - start
    rejected = total - passed

    print("\n" + "=" * 65)
    print("STAGE 1 RESULTS")
    print("=" * 65)
    print(f"Total processed            : {total:,}")
    print(f"Passed all filters         : {passed:,}  ({passed/total*100:.1f}%)")
    print(f"Rejected                   : {rejected:,}  ({rejected/total*100:.1f}%)")
    print(f"Time                       : {elapsed:.1f}s")
    print()
    print("Hard filter eliminations (in order):")
    for tag, count in sorted(filter_counts.items()):
        print(f"  {tag:<50} {count:>7,}")
    print()
    print("Passed candidates — soft signals:")
    print(f"  Location multiplier 0.35 applied (abroad, no relocate) : {location_penalized:>7,}")
    print(f"  Tier2 rescue (AI title + tier2 skills + retrieval desc) : {tier2_rescued:>7,}")
    print()
    print("Honeypot flags on passed candidates:")
    for flag, count in sorted(honeypot_counts.items()):
        print(f"  {flag:<50} {count:>7,}")
    print()
    print(f"Passed  → {passed_path}")
    print(f"Rejected→ {rejected_path}")
    print("=" * 65)

    if passed < 5_000:
        print(f"\n[WARN] Only {passed:,} passed — filters may be too aggressive.")
    if passed > 20_000:
        print(f"\n[WARN] {passed:,} passed — filters may be too lenient.")


def main():
    parser = argparse.ArgumentParser(description="Stage 1 hard filter pipeline")
    parser.add_argument(
        "--input", type=Path,
        default=Path("candidates.jsonl"),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path("./output"),
        help="Output directory",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("Stage 1 — Hard Filters + Honeypot Flags")
    print(f"Input : {args.input}")
    print(f"Output: {args.out_dir}")
    print()
    run(args.input, args.out_dir)


if __name__ == "__main__":
    main()