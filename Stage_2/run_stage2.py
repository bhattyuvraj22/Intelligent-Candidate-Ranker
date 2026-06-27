"""
Stage 2 — Scorer + Ranker
==========================
Reads stage1_passed.jsonl, scores every candidate using features.py,
applies protected shortlist logic, and outputs top 1,500.

Usage:
  python3 run_stage2.py --input output/stage1_passed.jsonl
                        --config stage2/stage2_config.yaml
                        --out_dir output/
"""
import argparse
import json
import time
from pathlib import Path

import yaml

from features import (
    get_career_temporal_weight,
    compute_skill_component,
    compute_career_component,
    compute_behavioral_component,
    compute_desc_depth_hits,
    compute_desc_depth_mult,
    compute_honeypot_mult,
    compute_location_penalty,
    compute_protected_score,
    is_protected_eligible,
    extract_signals,
)


def score_candidate(candidate, cfg):
    """
    Full scoring pipeline for one candidate.
    Returns enriched candidate dict with all scores attached.
    """
    # Temporal weight (used by tier1 depth scoring)
    temporal_weight = get_career_temporal_weight(candidate, cfg)

    # Three components
    skill_comp = compute_skill_component(candidate, cfg, temporal_weight)
    career_comp = compute_career_component(candidate, cfg)
    behavioral_comp = compute_behavioral_component(candidate, cfg)

    # Raw weighted blend
    w = cfg["component_weights"]
    raw_score = (
        w["skill"] * skill_comp
        + w["career"] * career_comp
        + w["behavioral"] * behavioral_comp
    )

    # Desc + honeypot multipliers (applied before location)
    desc_hits = compute_desc_depth_hits(candidate, cfg)
    desc_mult = compute_desc_depth_mult(desc_hits, cfg)
    honeypot_mult = compute_honeypot_mult(candidate, cfg)
    pre_location = raw_score * desc_mult * honeypot_mult

    # Location: additive penalty with floor.
    # Floor prevents compound stacking (honeypot + location) from burying
    # exceptional abroad candidates below mediocre local ones.
    # floor = pre_location * location_floor_factor (config, default 0.60)
    location_penalty = compute_location_penalty(candidate, cfg)
    floor_factor = float(cfg.get("location_floor_factor", 0.60))
    floor_score = pre_location * floor_factor
    final_score = max(floor_score, pre_location - location_penalty)

    # Protected shortlist: gate + skill/career-only score
    # desc_hits already computed above — reuse, no re-scan
    protected = is_protected_eligible(candidate, cfg, desc_hits)
    protected_sc = compute_protected_score(candidate, cfg, skill_comp, career_comp)

    # Signal extraction for reasoning
    signals = extract_signals(candidate, cfg)

    candidate["_s2"] = {
        "skill_component": round(skill_comp, 4),
        "career_component": round(career_comp, 4),
        "behavioral_component": round(behavioral_comp, 4),
        "raw_score": round(raw_score, 4),
        "desc_hits": desc_hits,
        "desc_mult": round(desc_mult, 4),
        "honeypot_mult": round(honeypot_mult, 4),
        "location_penalty": round(location_penalty, 4),
        "pre_location_score": round(pre_location, 4),
        "final_score": round(final_score, 6),
        "temporal_weight": round(temporal_weight, 4),
        "protected_eligible": protected,
        "protected_score": round(protected_sc, 4),
        "signals": signals,
    }
    return candidate


def run(input_path, config_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_size = cfg["stage2_output_size"]
    protected_size = cfg["protected_shortlist_size"]

    print(f"Stage 2 — Structured Feature Scorer")
    print(f"Input : {input_path}")
    print(f"Config: {config_path}")
    print(f"Output: {out_dir}")
    print()

    all_candidates = []
    total = 0
    start = time.time()

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            candidate = json.loads(line)
            scored = score_candidate(candidate, cfg)
            all_candidates.append(scored)

            if total % 2000 == 0:
                elapsed = time.time() - start
                print(f"  Scored {total:,} | {elapsed:.1f}s")

    elapsed = time.time() - start
    print(f"  Scored {total:,} total | {elapsed:.1f}s")
    print()

    # ── Protected shortlist ──────────────────────────────────────────
    # Gate: tier1_count >= 2 AND desc_hits >= 3 (~3% of pool).
    # Sorted by protected_score (skill+career only — no behavioral).
    # Purpose: guarantee inclusion for high-skill candidates who may have
    # low behavioral scores (haven't been active lately, slow responder).
    protected = [c for c in all_candidates if c["_s2"]["protected_eligible"]]
    protected.sort(key=lambda c: c["_s2"]["protected_score"], reverse=True)
    protected_top = protected[:protected_size]
    protected_ids = {c["candidate_id"] for c in protected_top}

    print(f"Protected eligible : {len(protected):,} | Taking top {len(protected_top)}")

    # ── Main ranking ─────────────────────────────────────────────────
    # All remaining candidates sorted by final_score (includes behavioral).
    remaining = [c for c in all_candidates if c["candidate_id"] not in protected_ids]
    remaining.sort(key=lambda c: c["_s2"]["final_score"], reverse=True)

    slots_needed = output_size - len(protected_top)
    main_top = remaining[:slots_needed]

    # ── Merge + deduplicate ──────────────────────────────────────────
    # Some protected candidates will naturally have high final_scores too
    # (strong skill AND strong behavioral). Dedup is a safety net.
    final_pool = protected_top + main_top
    seen = set()
    deduped = []
    for c in final_pool:
        if c["candidate_id"] not in seen:
            seen.add(c["candidate_id"])
            deduped.append(c)

    # ── Sort merged pool by final_score for Stage 3 ordering signal ──
    deduped.sort(key=lambda c: c["_s2"]["final_score"], reverse=True)
    top_1500 = deduped[:output_size]

    # ── Score distribution report ────────────────────────────────────
    import statistics
    scores = [c["_s2"]["final_score"] for c in top_1500]
    skill_scores = [c["_s2"]["skill_component"] for c in all_candidates]
    career_scores = [c["_s2"]["career_component"] for c in all_candidates]
    behavioral_scores = [c["_s2"]["behavioral_component"] for c in all_candidates]

    sorted_scores = sorted(scores, reverse=True)
    p10_idx = max(0, int(0.10 * len(sorted_scores)) - 1)
    p50_idx = max(0, int(0.50 * len(sorted_scores)) - 1)

    print("Score distribution (top 1,500):")
    print(f"  final_score : max={max(scores):.4f}  p10={sorted_scores[p10_idx]:.4f}  "
          f"p50={sorted_scores[p50_idx]:.4f}  min={min(scores):.4f}")
    print()
    print("Component distributions (all candidates):")
    print(f"  skill      : mean={statistics.mean(skill_scores):.3f}  median={statistics.median(skill_scores):.3f}  "
          f"std={statistics.stdev(skill_scores):.3f}")
    print(f"  career     : mean={statistics.mean(career_scores):.3f}  median={statistics.median(career_scores):.3f}  "
          f"std={statistics.stdev(career_scores):.3f}")
    print(f"  behavioral : mean={statistics.mean(behavioral_scores):.3f}  median={statistics.median(behavioral_scores):.3f}  "
          f"std={statistics.stdev(behavioral_scores):.3f}")
    print()

    # Location rescue audit: how many abroad candidates made top 1500
    abroad_rescued = sum(
        1 for c in top_1500
        if (c["profile"].get("country") or "").lower() != "india"
    )
    abroad_with_penalty = sum(
        1 for c in top_1500
        if c["_s2"]["location_penalty"] >= 0.12
    )

    # ── Write output ─────────────────────────────────────────────────
    out_path = out_dir / "stage2_scored.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for c in top_1500:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Sanity checks
    ela_found = any(c["candidate_id"] == "CAND_0000031" for c in top_1500)
    rescue_found = any(c["candidate_id"] == "CAND_0006833" for c in top_1500)

    print("=" * 60)
    print("STAGE 2 RESULTS")
    print("=" * 60)
    print(f"Total scored          : {total:,}")
    print(f"Protected eligible    : {len(protected):,}  ({len(protected)/total*100:.1f}%)")
    print(f"Protected shortlist   : {len(protected_top)}")
    print(f"Main ranking top      : {len(main_top)}")
    print(f"Final pool            : {len(top_1500)}")
    print(f"Abroad in top 1500    : {abroad_rescued}  (location_penalty>=0.12: {abroad_with_penalty})")
    print(f"Time                  : {time.time() - start:.1f}s")
    print()
    print("Sanity checks:")
    print(f"  Ela Singh  (CAND_0000031) in top 1500 : {ela_found}")
    print(f"  Tier2 rescue (CAND_0006833) in top 1500: {rescue_found}")
    print()
    print(f"Output → {out_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 scorer")
    parser.add_argument("--input", type=Path,
        default=Path("output/stage1_passed.jsonl"))
    parser.add_argument("--config", type=Path,
        default=Path("Stage_2/stage2_config.yaml"))
    parser.add_argument("--out_dir", type=Path,
        default=Path("./output"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}")
        raise SystemExit(1)

    run(args.input, args.config, args.out_dir)


if __name__ == "__main__":
    main()