"""
Stage 1 — Hard Filter Rule Engine
Applies 7 binary filters sequentially. First failure = candidate rejected.
Returns (passed: bool, reason: str | None) per candidate.

Design principles:
- Thin filters only: eliminates true binary disqualifiers, NOT soft signals
- Soft signals (boilerplate, salary inversion, job-hopping) go to Stage 4 multipliers
- When in doubt → PASS (false negatives are fatal; false positives are recoverable)
"""
from utils import (
    SERVICES_INDUSTRIES,
    NON_PRODUCT_INDUSTRIES,
    days_since,
    has_tier1_skill,
    has_retrieval_in_descriptions,
    is_non_tech_title,
    has_product_experience,
    all_ch_industries,
)


# ── F1: Location ─────────────────────────────────────────────────────────────

def filter_location(candidate: dict) -> tuple[bool, str | None]:
    """
    PASS if: country == India OR willing_to_relocate == True
    OUT  if: abroad AND not willing to relocate
    """
    profile = candidate["profile"]
    sig = candidate.get("redrob_signals", {})

    country = profile.get("country", "")
    relocate = sig.get("willing_to_relocate", False)

    if country == "India":
        return True, None
    if relocate:
        return True, None
    return False, f"F1:location | country={country} | relocate={relocate}"


# ── F2: Consulting-Only Career ────────────────────────────────────────────────

def filter_consulting_only(candidate: dict) -> tuple[bool, str | None]:
    """
    OUT if entire career_history is at services/consulting firms only.
    One product stint anywhere = PASS.
    Edge: empty career_history → PASS (can't confirm consulting-only).
    """
    industries = all_ch_industries(candidate)
    if not industries:
        return True, None  # No history → don't punish, check in Stage 4

    non_service = [i for i in industries if i not in SERVICES_INDUSTRIES]
    if non_service:
        return True, None
    return False, f"F2:consulting_only | industries={industries}"


# ── F3: Years of Experience Band ─────────────────────────────────────────────

def filter_experience_band(candidate: dict, min_years: float = 3, max_years: float = 15) -> tuple[bool, str | None]:
    """
    OUT if YoE < 3 (too junior) or > 15 (over-specialized / senior mismatch).
    Uses profile.years_of_experience directly — reported field, not computed.
    Pain point note: old script used max=9, changed to 15 per dataset analysis.
    """
    yoe = candidate["profile"].get("years_of_experience")
    if yoe is None:
        return True, None  # Missing → don't eliminate; penalize in Stage 4
    if yoe < min_years:
        return False, f"F3:yoe_too_low | yoe={yoe}"
    if yoe > max_years:
        return False, f"F3:yoe_too_high | yoe={yoe}"
    return True, None


# ── F4: Zero JD-Relevant Skills ──────────────────────────────────────────────

def filter_zero_relevant_skills(candidate: dict) -> tuple[bool, str | None]:
    """
    OUT if zero TIER1_SKILLS present in skills list.
    Report: 72,294 candidates have zero tier1 skills — fast elimination.
    Note: skill presence is weak signal; skill ABSENCE is a hard disqualifier.
    """
    if has_tier1_skill(candidate):
        return True, None
    return False, "F4:zero_tier1_skills"


# ── F5: Non-Technical Title + Zero Retrieval in Descriptions ─────────────────

def filter_nontechnical_no_retrieval(candidate: dict) -> tuple[bool, str | None]:
    """
    OUT if: current_title is non-technical AND no retrieval keywords in ANY description.
    PASS if: title is non-tech BUT descriptions show real retrieval/ML work.

    This catches keyword stuffers (5,420 in dataset) while preserving
    "Software Engineer at Swiggy" whose title doesn't say ML but descriptions do.

    CRITICAL: checks descriptions, NOT skills (descriptions harder to fake).
    """
    if not is_non_tech_title(candidate):
        return True, None  # Technical title → no issue

    # Non-tech title: check if descriptions redeem them
    if has_retrieval_in_descriptions(candidate):
        return True, None  # Non-tech title but real retrieval work → PASS

    title = candidate["profile"].get("current_title", "")
    return False, f"F5:nontechnical_no_retrieval | title={title}"


# ── F6: Zero Product Company Experience ──────────────────────────────────────

def filter_zero_product_experience(candidate: dict) -> tuple[bool, str | None]:
    """
    OUT if entire career is at non-product companies
    (services + manufacturing + conglomerate + paper products).
    
    Spot-checked: correctly cuts .NET @ Cognizant, QA @ Wipro/Manufacturing.
    Does NOT cut Ela Singh (Food Delivery, AI/ML, Transportation).
    """
    industries = all_ch_industries(candidate)
    if not industries:
        return True, None  # No history → PASS, penalize in Stage 4

    if has_product_experience(candidate):
        return True, None
    return False, f"F6:zero_product_exp | industries={industries}"


# ── F7: Behaviorally Dead ─────────────────────────────────────────────────────

def filter_behaviorally_dead(
    candidate: dict,
    max_days_inactive: int = 180,
    min_rr: float = 0.20,
    absolute_floor_rr: float = 0.10,
) -> tuple[bool, str | None]:
    """
    OUT if:
      - last_active > 180 days AND response_rate < 0.20 (both required), OR
      - response_rate < 0.10 regardless of last_active (completely unreachable)

    Report: good-fit candidates avg RR=0.55 vs 0.44 overall.
    17,245 candidates have RR < 0.20 — effectively unreachable.
    No candidate active <30 days (dataset artifact) — 30-90d = "active" tier.
    """
    sig = candidate.get("redrob_signals", {})
    rr = sig.get("recruiter_response_rate") or 0.0
    last_active_days = days_since(sig.get("last_active_date"))

    # Hard floor: completely unreachable regardless of activity
    if rr < absolute_floor_rr:
        return False, f"F7:rr_absolute_floor | rr={rr:.2f}"

    # Combination: inactive + unresponsive
    if last_active_days > max_days_inactive and rr < min_rr:
        return False, f"F7:behaviorally_dead | inactive={last_active_days}d | rr={rr:.2f}"

    return True, None


# ── Master filter runner ──────────────────────────────────────────────────────

def apply_all_filters(candidate: dict) -> tuple[bool, str | None]:
    """
    Run all 7 filters in order. First failure → return immediately.
    Returns (passed, reason_or_none).
    """
    checks = [
        filter_location,
        filter_consulting_only,
        filter_experience_band,
        filter_zero_relevant_skills,
        filter_nontechnical_no_retrieval,
        filter_zero_product_experience,
        filter_behaviorally_dead,
    ]
    for check in checks:
        passed, reason = check(candidate)
        if not passed:
            return False, reason
    return True, None
