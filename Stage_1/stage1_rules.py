"""
Stage 1 — Hard Kill Rules
=========================
Binary, deterministic, config-driven disqualifier checks.
Every candidate gets a full reason trail (for debugging + Stage 5 interview defense),
even if not rejected — soft flags are recorded but don't remove the candidate.

Design choice: plain Python + dict/list ops, no pandas.
Reason: per-record nested logic (dates, lists-of-dicts) maps directly onto
Python loops; pandas vectorization adds overhead without benefit at this
record structure, and this stage must stay fast/simple to keep Stage 1
near-zero cost in the 5-min budget.
"""
import json
import yaml
from pathlib import Path
from utils import parse_date, months_between, collect_candidate_text, all_companies

BASE = Path(__file__).resolve().parent.parent


def load_config(path=BASE / "config" / "disqualifiers.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Individual rule checks ────────────────────────────────────────────
# Each returns (triggered: bool, reason: str) or None if not applicable.

def check_timeline_duration_mismatch(candidate, cfg):
    tol = cfg["timeline"]["duration_tolerance_months"]
    for ch in candidate.get("career_history", []):
        start = parse_date(ch.get("start_date"))
        end = parse_date(ch.get("end_date"))
        stated = ch.get("duration_months")
        if start is None or stated is None:
            continue
        actual = months_between(start, end)
        if actual is not None and abs(actual - stated) > tol:
            return True, (
                f"TIMELINE_MISMATCH: '{ch.get('title')}' at '{ch.get('company')}' "
                f"stated duration_months={stated} but dates imply ~{actual} "
                f"(tolerance={tol})"
            )
    return False, None


def check_experience_tenure_overshoot(candidate, cfg):
    overshoot_pct = cfg["timeline"]["experience_overshoot_pct"]
    yoe = candidate["profile"].get("years_of_experience")
    if yoe is None:
        return False, None
    total_months = sum(
        ch.get("duration_months", 0) for ch in candidate.get("career_history", [])
    )
    allowed_max = yoe * 12 * (1 + overshoot_pct / 100)
    if total_months > allowed_max:
        return True, (
            f"EXPERIENCE_OVERSHOOT: career_history sums to {total_months} months "
            f"but years_of_experience={yoe} (~{yoe*12:.0f} mo) allows max "
            f"{allowed_max:.0f} mo (+{overshoot_pct}% tolerance)"
        )
    return False, None


def check_consulting_only(candidate, cfg):
    if not cfg["consulting_only"]["hard_reject"]:
        return False, None
    blocklist = {c.lower() for c in cfg["consulting_only"]["blocklist_companies"]}
    companies = all_companies(candidate)
    if not companies:
        return False, None
    if all(c.lower() in blocklist for c in companies):
        return True, (
            f"CONSULTING_ONLY: entire career history is within blocklist "
            f"{sorted(set(companies))} — no product-company experience"
        )
    return False, None


def check_zero_tech_signal(candidate, cfg):
    if not cfg["zero_tech_signal"]["hard_reject"]:
        return False, None
    keywords = cfg["zero_tech_signal"]["technical_keywords"]
    text = collect_candidate_text(candidate)
    if not any(kw in text for kw in keywords):
        return True, (
            "ZERO_TECH_SIGNAL: no technical/AI-adjacent keyword found in "
            "headline, summary, titles, or skills"
        )
    return False, None


def check_location_mismatch(candidate, cfg):
    target_cities = [c.lower() for c in cfg["location"]["target_cities"]]
    profile = candidate["profile"]
    country = (profile.get("country") or "").lower()
    location = (profile.get("location") or "").lower()
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    in_target = any(city in location for city in target_cities)
    if country != "india" and not willing and not in_target:
        triggered = cfg["location"]["hard_reject"]
        return triggered, (
            f"LOCATION_MISMATCH: country='{profile.get('country')}', "
            f"location='{profile.get('location')}', willing_to_relocate={willing} "
            f"({'HARD' if triggered else 'SOFT'} per config)"
        )
    return False, None


def check_job_hopping(candidate, cfg):
    threshold_months = cfg["job_hopping"]["short_stint_months"]
    min_stints = cfg["job_hopping"]["min_short_stints_to_flag"]
    short_stints = [
        ch for ch in candidate.get("career_history", [])
        if not ch.get("is_current") and (ch.get("duration_months") or 999) < threshold_months
    ]
    if len(short_stints) >= min_stints:
        triggered = cfg["job_hopping"]["hard_reject"]
        return triggered, (
            f"JOB_HOPPING: {len(short_stints)} stints under {threshold_months} months "
            f"(threshold={min_stints}) ({'HARD' if triggered else 'SOFT'} per config)"
        )
    return False, None


def check_experience_band(candidate, cfg):
    yoe = candidate["profile"].get("years_of_experience")
    lo, hi = cfg["experience_band"]["min_years"], cfg["experience_band"]["max_years"]
    if yoe is not None and not (lo <= yoe <= hi):
        triggered = cfg["experience_band"]["hard_reject"]
        return triggered, (
            f"EXPERIENCE_BAND: years_of_experience={yoe} outside [{lo}, {hi}] "
            f"({'HARD' if triggered else 'SOFT'} per config)"
        )
    return False, None


RULES = [
    check_timeline_duration_mismatch,
    check_experience_tenure_overshoot,
    check_consulting_only,
    check_zero_tech_signal,
    check_location_mismatch,
    check_job_hopping,
    check_experience_band,
]


def evaluate_candidate(candidate, cfg):
    """Run all rules. Returns dict with status + full reason trail."""
    hard_rejected = False
    reasons = []
    for rule_fn in RULES:
        triggered, reason = rule_fn(candidate, cfg)
        if reason:
            reasons.append(reason)
        if triggered:
            hard_rejected = True
    return {
        "candidate_id": candidate["candidate_id"],
        "status": "REJECTED" if hard_rejected else "PASS",
        "reasons": reasons,
    }


def run_stage1(input_path, config_path=None, output_path=None):
    cfg = load_config(config_path) if config_path else load_config()
    candidates = json.load(open(input_path))
    results = [evaluate_candidate(c, cfg) for c in candidates]

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    passed = [r for r in results if r["status"] == "PASS"]
    rejected = [r for r in results if r["status"] == "REJECTED"]
    print(f"Total: {len(results)} | PASS: {len(passed)} | REJECTED: {len(rejected)}")
    return results


if __name__ == "__main__":
    run_stage1(
        input_path=BASE / "data" / "sample_candidates.json",
        output_path=BASE / "output" / "stage1_results.json",
    )
