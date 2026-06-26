"""
Stage 1 — Hard Filter + Honeypot Flag Engine
Final version (v4).

Hard eliminates (first failure = OUT):
  F1  — Location: abroad + not willing to relocate + no tier1 skill
  F2  — Consulting-only career
  F3  — YoE out of range (< 3 or > 15)
  F4  — Zero tier1 skills (with tier2 rescue for genuine edge cases)
  F5  — Non-technical title + zero retrieval in descriptions
  F6  — Zero product company experience
  F7  — Behaviorally dead (inactive + unresponsive)
  H1  — >50% skills have impossible duration vs YoE
  H3  — Job started >=3 years before graduation (fabricated timeline)
  H4  — Total career months impossibly exceed YoE

Soft flags (PASS but _honeypot_flags attached for Stage 4 penalty):
  H1-soft — some (not majority) impossible skill durations
  H5      — salary inversion (too common to hard-eliminate)

Design: when in doubt → PASS. False negatives are fatal; false positives
are penalized and recoverable downstream.
"""
from utils import (
    SERVICES_INDUSTRIES,
    NON_PRODUCT_INDUSTRIES,
    TIER1_SKILLS,
    NON_TECH_TITLE_KEYWORDS,
    days_since,
    has_tier1_skill,
    has_tier2_rescue,
    has_retrieval_in_descriptions,
    is_non_tech_title,
    has_product_experience,
    all_ch_industries,
    get_earliest_graduation_year,
    get_earliest_job_start_year,
)


# ════════════════════════════════════════════════════════════════
# HARD FILTERS
# ════════════════════════════════════════════════════════════════

def filter_location(candidate):
    """
    PASS: India | abroad + willing to relocate | abroad + tier1 skill
    OUT:  abroad + not willing to relocate + no tier1 skill

    No multipliers here — location scoring handled in Stage 4.
    """
    profile = candidate["profile"]
    sig = candidate.get("redrob_signals", {})
    country = profile.get("country", "")
    relocate = sig.get("willing_to_relocate", False)

    if country == "India":
        return True, None
    if relocate:
        return True, None
    if has_tier1_skill(candidate):
        return True, None
    return False, f"F1:abroad_no_relocate_no_tier1 | country={country}"


def filter_consulting_only(candidate):
    """OUT if entire career is at services/consulting firms. One product stint = PASS."""
    industries = all_ch_industries(candidate)
    if not industries:
        return True, None
    if any(i not in SERVICES_INDUSTRIES for i in industries):
        return True, None
    return False, f"F2:consulting_only | industries={industries}"


def filter_experience_band(candidate, min_years=3.0, max_years=15.0):
    """OUT if YoE < 3 or > 15. Missing YoE → PASS (penalize in Stage 4)."""
    yoe = candidate["profile"].get("years_of_experience")
    if yoe is None:
        return True, None
    if yoe < min_years:
        return False, f"F3:yoe_too_low | yoe={yoe}"
    if yoe > max_years:
        return False, f"F3:yoe_too_high | yoe={yoe}"
    return True, None


def filter_zero_relevant_skills(candidate):
    """
    OUT if zero tier1 skills AND tier2 rescue condition not met.
    Tier2 rescue: AI title + product exp + 2+ tier2 skills + retrieval in desc.
    Validated: rescues exactly CAND_0006833 (AI Research Engineer).
    """
    if has_tier1_skill(candidate):
        return True, None
    if has_tier2_rescue(candidate):
        candidate["_tier2_rescue"] = True
        return True, None
    return False, "F4:zero_tier1_skills"


def filter_nontechnical_no_retrieval(candidate):
    """
    OUT if non-technical title AND no retrieval keywords in any career description.
    PASS if non-tech title but descriptions show real retrieval/ML work.
    """
    if not is_non_tech_title(candidate):
        return True, None
    if has_retrieval_in_descriptions(candidate):
        return True, None
    title = candidate["profile"].get("current_title", "")
    return False, f"F5:nontechnical_no_retrieval | title={title}"


def filter_zero_product_experience(candidate):
    """OUT if entire career at non-product companies."""
    industries = all_ch_industries(candidate)
    if not industries:
        return True, None
    if has_product_experience(candidate):
        return True, None
    return False, f"F6:zero_product_exp | industries={industries}"


def filter_behaviorally_dead(candidate, max_days_inactive=180, min_rr=0.20, absolute_floor_rr=0.10):
    """
    OUT if: (last_active > 180d AND RR < 0.20) OR (RR < 0.10 regardless).
    Both conditions required for main check — prevents over-elimination.
    """
    sig = candidate.get("redrob_signals", {})
    rr = sig.get("recruiter_response_rate") or 0.0
    last_active_days = days_since(sig.get("last_active_date"))

    if rr < absolute_floor_rr:
        return False, f"F7:rr_absolute_floor | rr={rr:.2f}"
    if last_active_days > max_days_inactive and rr < min_rr:
        return False, f"F7:behaviorally_dead | inactive={last_active_days}d | rr={rr:.2f}"
    return True, None


def filter_h1_impossible_skills_hard(candidate):
    """
    OUT if >50% of skills have duration_months > YoE × 12.
    Threshold >50% avoids punishing minor date entry errors.
    """
    yoe = candidate["profile"].get("years_of_experience") or 0
    yoe_months = yoe * 12
    if yoe_months <= 0:
        return True, None

    skills = candidate.get("skills", [])
    if not skills:
        return True, None

    impossible = sum(1 for s in skills if (s.get("duration_months") or 0) > yoe_months)
    if impossible / len(skills) > 0.50:
        return False, f"H1:impossible_skill_durations | {impossible}/{len(skills)} exceed YoE ({yoe}yr)"
    return True, None


def filter_h3_job_before_graduation(candidate):
    """
    Split rule — same pattern as F1 location:
      job_before_graduation AND has tier1 skill → PASS with H3 soft flag
      job_before_graduation AND no tier1 skill  → HARD OUT

    Data shows 187 candidates with tier1 skills + job_before_graduation (gap>=3yr).
    These include AI Research Engineers, Senior Data Engineers, Search Engineers
    with strong retrieval skills — cannot afford to lose them.

    The no-tier1 group (0 in current data) are pure noise with fabricated timelines.
    H3 flag on passed candidates → Stage 4 applies x0.70 penalty (1 flag).

    Gap threshold: >= 3 years (2yr tolerance rescues legitimate internship cases).
    Education schema uses integer end_year fields (not date strings).
    """
    grad_year = get_earliest_graduation_year(candidate)
    job_year = get_earliest_job_start_year(candidate)

    if grad_year is None or job_year is None:
        return True, None

    gap = grad_year - job_year
    if gap < 3:
        return True, None  # Within tolerance — no issue

    # gap >= 3yr: check if tier1 skill redeems them
    if has_tier1_skill(candidate):
        candidate.setdefault('_honeypot_flags', []).append('H3_job_before_graduation')
        return True, None

    return False, f"H3:job_before_graduation_no_tier1 | gap={gap}yr job={job_year} grad={grad_year}"


def filter_h4_impossible_career_timeline(candidate):
    """
    OUT if total career months > YoE × 12 + 36.
    36-month buffer handles legitimate overlapping/part-time roles.
    """
    yoe = candidate["profile"].get("years_of_experience") or 0
    yoe_months = yoe * 12
    if yoe_months <= 0:
        return True, None

    total_career = sum(ch.get("duration_months") or 0 for ch in candidate.get("career_history", []))
    if total_career > yoe_months + 36:
        return False, f"H4:impossible_career_total | career={total_career}mo YoE={yoe}yr"
    return True, None


# ════════════════════════════════════════════════════════════════
# SOFT HONEYPOT FLAGS — PASS but flagged for Stage 4 penalty
# ════════════════════════════════════════════════════════════════

def compute_honeypot_flags(candidate):
    """
    Attaches _honeypot_flags list to candidate for Stage 4 multipliers.
      2+ flags → × 0.50
      1 flag   → × 0.70
      0 flags  → × 1.00

    H3 removed from soft flags — now a hard eliminate (gap >= 3yr).
    H6 kept but fires 0 times in practice (stuffers caught by F4/F5 first).
    """
    flags = list(candidate.get("_honeypot_flags", []))  # preserve flags set by hard filters (e.g. H3)
    yoe = candidate["profile"].get("years_of_experience") or 0
    yoe_months = yoe * 12
    sig = candidate.get("redrob_signals", {})

    # H1-soft: some (not majority) impossible skill durations
    if yoe_months > 0:
        skills = candidate.get("skills", [])
        impossible = sum(1 for s in skills if (s.get("duration_months") or 0) > yoe_months)
        total = len(skills)
        if total > 0 and 0 < impossible / total <= 0.50:
            flags.append("H1_soft_skill_duration")

    # H5: salary inversion — too common (18,865 cases) to hard-eliminate
    # but meaningful in combination with other flags
    sal = sig.get("expected_salary_range_inr_lpa", {})
    if sal and isinstance(sal, dict):
        sal_min = sal.get("min") or 0
        sal_max = sal.get("max") or 0
        if sal_min > sal_max > 0:
            flags.append("H5_salary_inversion")

    # H6: non-tech title + 2+ expert tier1 skills (canonical stuffer pattern)
    title = candidate["profile"].get("current_title", "").lower()
    if any(nt in title for nt in NON_TECH_TITLE_KEYWORDS):
        expert_tier1 = sum(
            1 for s in candidate.get("skills", [])
            if s.get("name") in TIER1_SKILLS and s.get("proficiency") == "expert"
        )
        if expert_tier1 >= 2:
            flags.append("H6_stuffer_expert_tier1")

    candidate["_honeypot_flags"] = flags
    return flags


# ════════════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════════════

def apply_all_filters(candidate):
    """
    Run all hard filters sequentially — first failure = OUT.
    If all pass: compute soft honeypot flags (attached to candidate dict).
    Returns (passed: bool, reason: str | None).
    """
    hard_filters = [
        filter_location,
        filter_consulting_only,
        filter_experience_band,
        filter_zero_relevant_skills,
        filter_nontechnical_no_retrieval,
        filter_zero_product_experience,
        filter_behaviorally_dead,
        filter_h1_impossible_skills_hard,
        filter_h3_job_before_graduation,
        filter_h4_impossible_career_timeline,
    ]

    for check in hard_filters:
        passed, reason = check(candidate)
        if not passed:
            return False, reason

    compute_honeypot_flags(candidate)
    return True, None