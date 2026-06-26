"""
Stage 1 Runner — Hard Filter Pipeline
Streams candidates.jsonl line by line (no full load into memory).
Outputs:
  - stage1_passed.jsonl  : candidates surviving all 7 filters
  - stage1_rejected.jsonl: rejected candidates with filter reason tag

Usage:
  python run_stage1.py --input candidates.jsonl --out_dir ./output
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

    total = 0
    passed = 0
    filter_counts = Counter()
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
                print(f"[WARN] Skipping malformed JSON line {total}: {e}", file=sys.stderr)
                continue

            ok, reason = apply_all_filters(candidate)

            if ok:
                fpass.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                passed += 1
            else:
                # Tag the record with rejection reason for audit trail
                candidate["_stage1_rejected"] = reason
                frej.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                # Extract filter tag (e.g. "F1:location") for counter
                tag = reason.split("|")[0].strip() if reason else "unknown"
                filter_counts[tag] += 1

            # Progress every 10K
            if total % 10_000 == 0:
                elapsed = time.time() - start
                print(f"  Processed {total:,} | passed so far: {passed:,} | {elapsed:.1f}s")

    elapsed = time.time() - start
    rejected = total - passed

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 RESULTS")
    print("=" * 60)
    print(f"Total candidates processed : {total:,}")
    print(f"Passed all filters         : {passed:,}  ({passed/total*100:.1f}%)")
    print(f"Rejected                   : {rejected:,}  ({rejected/total*100:.1f}%)")
    print(f"Time elapsed               : {elapsed:.1f}s")
    print()
    print("Eliminated per filter (in order applied):")
    for tag, count in sorted(filter_counts.items()):
        print(f"  {tag:<45} {count:>7,}")
    print()
    print(f"Output — passed  : {passed_path}")
    print(f"Output — rejected: {rejected_path}")
    print("=" * 60)

    # Sanity check: warn if passed count is wildly off expected
    if passed < 5_000:
        print(f"\n[WARN] Only {passed:,} passed — filters may be too aggressive. Review rejected sample.")
    if passed > 20_000:
        print(f"\n[WARN] {passed:,} passed — filters may be too lenient. Review rejected sample.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 hard filter pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/claude/challenge_data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./output"),
        help="Output directory for passed/rejected files",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Stage 1 — Hard Filters")
    print(f"Input : {args.input}")
    print(f"Output: {args.out_dir}")
    print()

    run(args.input, args.out_dir)


if __name__ == "__main__":
    main()
