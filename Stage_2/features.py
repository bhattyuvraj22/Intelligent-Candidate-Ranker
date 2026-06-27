"""
Stage 2 — Feature Computation (FINAL)
======================================
19 features across 3 groups + 3 multipliers.
All values validated against actual passed pool (11,415 candidates).
Self-contained — no imports from Stage 1.

All 9 loophole fixes applied + 4 additional data-validated corrections:
  FIX 1: Temporal decay via career-position, not description text matching
  FIX 2: ML_TITLES / ML_INDUSTRIES defined in config
  FIX 3: Duration inflation removed from honeypot flags (only credibility_mult)
  FIX 4: tier1_max_possible = 2.75 (data-validated p99, not theoretical 3.75)
  FIX 5: Two keyword lists — normalized for skills, raw aliases for descriptions
  FIX 6: Protected shortlist gate uses OR (ml_yoe>0 OR desc_hits>=3)
  FIX 7: search_norm weight at 10% (circularity risk)
  FIX 8: Career momentum defaults to 0.5 for single-role candidates
  FIX 9: Location multiplier computed fresh here, not from Stage 1 flags
  FIX 10: search_saturation=500 (not 350) — good-fit mean=471 in data
  FIX 11: saved_saturation=27 (not 20) — good-fit median=27 in data
  FIX 12: notice_period default=0.35 for any value >90d (not >180d gap)
  FIX 13: Honeypot flag 3 fires 0 times in data — kept but noted as dead signal
"""
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 26)


# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════

def _parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _days_since(date_str):
    d = _parse_date(date_str)
    if d is None:
        return None
    return (REFERENCE_DATE - d).days


def _collect_description_text(candidate):
    """All career_history descriptions concatenated — lowercased."""
    parts = [ch.get("description", "") or "" for ch in candidate.get("career_history", [])]
    return " ".join(parts).lower()


def _normalize_skill_name(raw_name, aliases):
    """Map raw skill name to canonical Tier1/2 name using alias dict."""
    return aliases.get(raw_name.lower().strip(), raw_name)


def _get_tier_sets(cfg):
    """Build lowercased lookup sets for Tier1 and Tier2 skills."""
    tier1 = {s.lower() for s in cfg["tier1_skills"]}
    tier2 = {s.lower() for s in cfg["tier2_skills"]}
    return tier1, tier2


def _is_ml_role(role, ml_titles_set, ml_industries_set):
    """True if a career_history role is ML/AI relevant."""
    title = (role.get("title") or "").lower()
    industry = (role.get("industry") or "")
    return (
        any(t in title for t in ml_titles_set)
        or industry in ml_industries_set
    )


# ══════════════════════════════════════════════════════════════
# TEMPORAL WEIGHT
# ══════════════════════════════════════════════════════════════

def get_career_temporal_weight(candidate, cfg):
    """
    FIX 1: Temporal decay based on CAREER POSITION — not description text.
    Finds most recent ML/AI role and returns the appropriate decay weight.
    Single-role with ML title → current_role_is_ml weight.
    No ML role anywhere → no_ml_role_found weight (0.30).
    """
    ml_titles_set = {t.lower() for t in cfg["ml_titles"]}
    ml_industries_set = set(cfg["ml_industries"])
    decay = cfg["temporal_decay"]

    career = sorted(
        candidate.get("career_history", []),
        key=lambda r: r.get("start_date") or "0000-00-00",
        reverse=True,
    )

    for role in career:
        if not _is_ml_role(role, ml_titles_set, ml_industries_set):
            continue
        if role.get("is_current"):
            return decay["current_role_is_ml"]
        end = _parse_date(role.get("end_date"))
        if end is None:
            return decay["current_role_is_ml"]
        years_ago = (REFERENCE_DATE - end).days / 365.25
        if years_ago < 2:
            return decay["ml_role_under_2yr"]
        elif years_ago < 4:
            return decay["ml_role_2_to_4yr"]
        else:
            return decay["ml_role_over_4yr"]

    return decay["no_ml_role_found"]


# ══════════════════════════════════════════════════════════════
# COMPONENT 1 — SKILL EVIDENCE (45% of final score)
# ══════════════════════════════════════════════════════════════

def compute_tier1_depth(candidate, cfg, temporal_weight):
    """
    Trust-weighted Tier1 skill depth, decayed by career position.
    trust = proficiency_score × endorsement_ratio × duration_ratio
    Normalized by tier1_max_possible (FIX 4: 2.75, data-validated p99).

    Previous value 3.75 was theoretical — caused p99 candidate to score
    only 0.73, badly compressing discrimination. 2.75 lets p99 hit ~1.0.
    """
    tier1_set, _ = _get_tier_sets(cfg)
    aliases = cfg["skill_aliases"]
    expected = cfg["expected_by_proficiency"]
    prof_scores = cfg["proficiency_scores"]
    max_possible = cfg["tier1_max_possible"]

    total_trust = 0.0
    for skill in candidate.get("skills", []):
        canonical = _normalize_skill_name(skill.get("name", ""), aliases)
        if canonical.lower() not in tier1_set:
            continue

        prof = skill.get("proficiency", "beginner")
        endorsements = skill.get("endorsements", 0) or 0
        duration = skill.get("duration_months", 0) or 0

        prof_score = prof_scores.get(prof, 0.15)
        exp = expected.get(prof, expected["beginner"])

        endorse_trust = min(1.0, endorsements / exp["endorsements"]) if exp["endorsements"] > 0 else 0.0
        duration_trust = min(1.0, duration / exp["duration_months"]) if exp["duration_months"] > 0 else 0.0

        total_trust += prof_score * endorse_trust * duration_trust * temporal_weight

    return min(1.0, total_trust / max_possible) if max_possible > 0 else 0.0


def compute_tier2_presence(candidate, cfg):
    """
    Endorsement-adjusted count of Tier2 skills.
    Zero-endorsement claims count as 0.3 (not 1.0) — anti-stuffer.
    Saturates at 4 skills.
    """
    _, tier2_set = _get_tier_sets(cfg)
    aliases = cfg["skill_aliases"]
    saturation = cfg["tier2_saturation_count"]

    count = 0.0
    for skill in candidate.get("skills", []):
        canonical = _normalize_skill_name(skill.get("name", ""), aliases)
        if canonical.lower() not in tier2_set:
            continue
        endorse_bonus = 1.0 if (skill.get("endorsements", 0) or 0) > 0 else 0.3
        count += endorse_bonus

    return min(1.0, count / saturation) if saturation > 0 else 0.0


def compute_assessment_verification(candidate, cfg):
    """
    Redrob assessment scores for Tier1/2 skills.
    Data: 23.2% of passed pool have tier1-relevant assessments.
    Default 0.3 (not 0.0) — absence is not guilt.
    0.40 for assessments completed but not on relevant skills.
    """
    tier1_set, tier2_set = _get_tier_sets(cfg)
    aliases = cfg["skill_aliases"]

    assessment_scores_raw = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores") or {}
    )

    relevant = []
    for skill_name, score in assessment_scores_raw.items():
        canonical = _normalize_skill_name(skill_name, aliases)
        if canonical.lower() in tier1_set or canonical.lower() in tier2_set:
            relevant.append(score)

    if relevant:
        return sum(relevant) / len(relevant) / 100.0
    elif assessment_scores_raw:
        return 0.40   # completed assessments but not for relevant skills
    else:
        return 0.30   # no assessments at all


def compute_skill_credibility(candidate, cfg):
    """
    FIX 3: Multiplier for fabricated skill durations.
    Applied to entire skill_component ONLY.
    NOT included in honeypot flags (was double-penalizing same signal).
    Floor at 0.20 — never fully zeroes out skill evidence.
    """
    yoe = (candidate["profile"].get("years_of_experience") or 1.0)
    max_allowed = yoe * 12.0
    skills = candidate.get("skills", [])

    if not skills:
        return 1.0

    impossible = sum(
        1 for s in skills
        if (s.get("duration_months") or 0) > max_allowed
    )
    impossible_ratio = impossible / len(skills)
    return max(0.20, 1.0 - impossible_ratio * 0.80)


def compute_skill_component(candidate, cfg, temporal_weight):
    """Combine all skill sub-scores into final skill_component."""
    weights = cfg["skill_weights"]
    tier1 = compute_tier1_depth(candidate, cfg, temporal_weight)
    tier2 = compute_tier2_presence(candidate, cfg)
    assessment = compute_assessment_verification(candidate, cfg)
    credibility = compute_skill_credibility(candidate, cfg)

    raw = (
        weights["tier1"] * tier1
        + weights["tier2"] * tier2
        + weights["assessment"] * assessment
    )
    return raw * credibility


# ══════════════════════════════════════════════════════════════
# COMPONENT 2 — CAREER EVIDENCE (30% of final score)
# ══════════════════════════════════════════════════════════════

def compute_product_ratio(candidate, cfg):
    """
    Fraction of career months at product-industry companies.
    Data: mean=0.51, median=0.48 in passed pool.
    """
    product_industries = set(cfg["product_industries"])
    total = 0
    product = 0

    for role in candidate.get("career_history", []):
        dur = role.get("duration_months") or 0
        total += dur
        if (role.get("industry") or "") in product_industries:
            product += dur

    return product / total if total > 0 else 0.0


def compute_career_momentum(candidate, cfg):
    """
    Measures whether career is moving TOWARD or AWAY from ML.
    FIX 8: Single-role or insufficient history → 0.5 (neutral, not penalized).

    recent (<= 2yr ago): fraction of roles that are ML
    historical (> 2yr ago): fraction of roles that are ML
    momentum = recent_ratio - (historical_ratio × 0.5)
    score = clamp(0.5 + momentum, 0, 1)
    """
    ml_titles_set = {t.lower() for t in cfg["ml_titles"]}
    ml_industries_set = set(cfg["ml_industries"])
    career = candidate.get("career_history", [])

    if len(career) <= 1:
        return 0.5   # FIX 8: neutral for single-role

    recent_roles = []
    historical_roles = []

    for role in career:
        if role.get("is_current"):
            recent_roles.append(role)
            continue
        end = _parse_date(role.get("end_date"))
        if end is None:
            recent_roles.append(role)
            continue
        years_ago = (REFERENCE_DATE - end).days / 365.25
        if years_ago <= 2:
            recent_roles.append(role)
        else:
            historical_roles.append(role)

    def ml_ratio(roles):
        if not roles:
            return None
        ml_count = sum(1 for r in roles if _is_ml_role(r, ml_titles_set, ml_industries_set))
        return ml_count / len(roles)

    recent = ml_ratio(recent_roles)
    historical = ml_ratio(historical_roles)

    if recent is None or (recent is None and historical is None):
        return 0.5
    if historical is None:
        return 0.5 + (recent * 0.5)

    momentum = recent - (historical * 0.5)
    return max(0.0, min(1.0, 0.5 + momentum))


def compute_ml_yoe(candidate, cfg):
    """
    ML-specific months of experience.
    Saturates at 60 months (5 years).
    Data: ml_months mean=39, median=38 — good discrimination.
    """
    ml_titles_set = {t.lower() for t in cfg["ml_titles"]}
    ml_industries_set = set(cfg["ml_industries"])
    saturation = cfg["ml_yoe_saturation_months"]

    ml_months = sum(
        role.get("duration_months") or 0
        for role in candidate.get("career_history", [])
        if _is_ml_role(role, ml_titles_set, ml_industries_set)
    )
    return min(1.0, ml_months / saturation) if saturation > 0 else 0.0


def compute_title_relevance(candidate, cfg):
    """
    Current title scored against ordered relevance tiers.
    First match wins — prevents double-counting.
    Default 0.05 for unmatched/non-technical titles.
    """
    current_title = (candidate["profile"].get("current_title") or "").lower()

    for tier in cfg["title_relevance_tiers"]:
        if any(kw in current_title for kw in tier["keywords"]):
            return float(tier["score"])

    return 0.05


def compute_stability_edu(candidate, cfg):
    """
    Composite: 60% tenure stability + 40% best education tier.
    Tenure: avg past-role duration / 24 months (capped at 1.0).
    Education: best tier across all degrees.
    """
    past_roles = [r for r in candidate.get("career_history", []) if not r.get("is_current")]
    if past_roles:
        avg_tenure = sum(r.get("duration_months") or 0 for r in past_roles) / len(past_roles)
        tenure_score = min(1.0, avg_tenure / 24.0)
    else:
        tenure_score = 0.5   # neutral for new-to-market

    edu_scores_map = cfg["education_tier_scores"]
    best_edu = float(edu_scores_map.get("unknown", 0.35))

    for edu in candidate.get("education", []):
        tier = edu.get("tier", "unknown")
        score = float(edu_scores_map.get(tier, edu_scores_map.get("unknown", 0.35)))
        best_edu = max(best_edu, score)

    return 0.60 * tenure_score + 0.40 * best_edu


def compute_career_component(candidate, cfg):
    """Combine all career sub-scores into final career_component."""
    weights = cfg["career_weights"]
    return (
        weights["product_ratio"] * compute_product_ratio(candidate, cfg)
        + weights["momentum"] * compute_career_momentum(candidate, cfg)
        + weights["ml_yoe"] * compute_ml_yoe(candidate, cfg)
        + weights["title_relevance"] * compute_title_relevance(candidate, cfg)
        + weights["stability_edu"] * compute_stability_edu(candidate, cfg)
    )


# ══════════════════════════════════════════════════════════════
# COMPONENT 3 — BEHAVIORAL AVAILABILITY (25% of final score)
# ══════════════════════════════════════════════════════════════

def compute_behavioral_component(candidate, cfg):
    """
    6 behavioral signals from Redrob platform data.

    FIX 7: search_norm at 10% (down from proposed 20%) — circularity risk
    FIX 10: search_saturation=500 (good-fit mean=471)
    FIX 11: saved_saturation=27 (good-fit median=27)
    FIX 12: notice defaults to 0.35 for any value not in brackets (<= check)

    Recency formula anchored to real distribution (no candidate < 30d inactive):
    max(0, 1.0 - (days_inactive - 30) / 150)
    → 30d = 1.0, 180d = 0.0
    """
    signals = candidate.get("redrob_signals", {}) or {}
    weights = cfg["behavioral_weights"]

    # Response rate (direct 0–1 value)
    rr = float(signals.get("recruiter_response_rate") or 0.0)

    # Recency — anchored to 30d minimum (FIX: dataset has no <30d candidates)
    days_inactive = _days_since(signals.get("last_active_date"))
    if days_inactive is None:
        recency = 0.30
    else:
        recency = max(0.0, 1.0 - max(0, days_inactive - 30) / 150.0)

    # Interview completion rate (direct 0–1 value)
    icr = float(signals.get("interview_completion_rate") or 0.0)

    # Availability: notice period → score (interpolated) + open_to_work bonus
    notice = signals.get("notice_period_days")
    if notice is None:
        notice = 90
    notice_scores_raw = {int(k): float(v) for k, v in cfg["notice_period_scores"].items()}
    notice_score = 0.10   # FIX 12: default for anything > max bracket
    for threshold in sorted(notice_scores_raw.keys()):
        if notice <= threshold:
            notice_score = notice_scores_raw[threshold]
            break
    otw_bonus = 0.10 if signals.get("open_to_work_flag") else 0.0
    availability = min(1.0, notice_score + otw_bonus)

    # Search appearances — FIX 10: saturation=500 (data-validated)
    search = float(signals.get("search_appearance_30d") or 0)
    search_norm = min(1.0, search / cfg["search_saturation"])

    # Saved by recruiters — FIX 11: saturation=27 (data-validated)
    saved = float(signals.get("saved_by_recruiters_30d") or 0)
    saved_norm = min(1.0, saved / cfg["saved_saturation"])

    return (
        weights["response_rate"] * rr
        + weights["recency"] * recency
        + weights["interview"] * icr
        + weights["availability"] * availability
        + weights["search_norm"] * search_norm
        + weights["saved"] * saved_norm
    )


# ══════════════════════════════════════════════════════════════
# MULTIPLIERS
# ══════════════════════════════════════════════════════════════

def compute_desc_depth_hits(candidate, cfg):
    """
    FIX 5: Count UNIQUE retrieval keywords in description text.
    Uses raw alias list — NOT normalized skill names.
    Returns raw hit count (used in protected shortlist gate too).
    """
    raw_keywords = cfg["retrieval_keywords_raw"]
    desc_text = _collect_description_text(candidate)
    return len({kw for kw in raw_keywords if kw in desc_text})


def compute_desc_depth_mult(desc_hits, cfg):
    """
    Maps unique keyword hits → [0.85, 1.0].
    0 hits → 0.85 (mild penalty, not zero — absence could just be writing style).
    Saturates at desc_depth_saturation hits → 1.0.
    """
    saturation = cfg["desc_depth_saturation"]
    ratio = min(1.0, desc_hits / saturation) if saturation > 0 else 0.0
    return 0.85 + (0.15 * ratio)


def compute_honeypot_mult(candidate, cfg):
    """
    FIX 3: 3 honeypot flags only.
    Duration inflation REMOVED — handled by skill_credibility_mult.

    Flag 1: salary min > max (verified: 1,352 in passed pool)
    Flag 2: job start before graduation >1yr (verified: 639 in passed pool)
    Flag 3: expert + 0 endorsements + <=6 months (fires 0 times in data —
             kept as future-proof signal, costs nothing to check)
    """
    signals = candidate.get("redrob_signals", {}) or {}
    flags = 0

    # Flag 1: salary inversion
    sal = signals.get("expected_salary_range_inr_lpa") or {}
    sal_min = sal.get("min") or 0
    sal_max = sal.get("max") or 0
    if sal_min > 0 and sal_max > 0 and sal_min > sal_max:
        flags += 1

    # Flag 2: job started before graduation (>1yr tolerance)
    earliest_grad = None
    for edu in candidate.get("education", []):
        ey = edu.get("end_year")
        if ey and (earliest_grad is None or ey < earliest_grad):
            earliest_grad = ey

    if earliest_grad:
        for role in candidate.get("career_history", []):
            start = _parse_date(role.get("start_date"))
            if start and start.year < earliest_grad - 1:
                flags += 1
                break

    # Flag 3: expert claim with zero evidence (fires 0x in current data)
    for skill in candidate.get("skills", []):
        if (
            skill.get("proficiency") == "expert"
            and (skill.get("endorsements") or 0) == 0
            and (skill.get("duration_months") or 99) <= 6
        ):
            flags += 1
            break

    penalty_map = {int(k): float(v) for k, v in cfg["honeypot_penalty_map"].items()}
    capped_flags = min(flags, max(penalty_map.keys()))
    return penalty_map.get(capped_flags, 0.30)


def compute_location_penalty(candidate, cfg):
    """
    Additive penalty computed from raw profile fields (not Stage 1 flags).

    Why additive instead of multiplicative: the old 0.35 multiplier took a
    raw score of 0.90 → 0.315, dropping a 9-tier1-skill ML engineer below
    hundreds of mediocre India-based candidates. That is elimination, not a
    penalty. Additive capped at 0.18 means the same engineer lands at ≥0.72,
    still below a comparable India-Noida candidate (0.90) but not buried.

    Penalties (read from config):
      India premium  (noida/pune)    → 0.00
      India target   (hyd/blr/etc.)  → 0.03
      India other + willing relocate → 0.06
      India other, no relocate       → 0.10
      Abroad + willing to relocate   → 0.12
      Abroad + no relocate (tier1)   → 0.18   ← max possible

    Floor enforced in caller: final >= pre_location × location_floor_factor.
    """
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {}) or {}

    country = (profile.get("country") or "").lower().strip()
    location = (profile.get("location") or "").lower().strip()
    willing = bool(signals.get("willing_to_relocate", False))

    penalties = cfg["location_penalties"]
    premium_cities = cfg["premium_cities"]
    target_cities = cfg["target_cities"]

    in_premium = any(city in location for city in premium_cities)
    in_target = any(city in location for city in target_cities)

    if country == "india":
        if in_premium:
            return float(penalties["india_premium"])
        elif in_target:
            return float(penalties["india_target"])
        elif willing:
            return float(penalties["india_other_willing"])
        else:
            return float(penalties["india_other"])
    else:
        return float(penalties["abroad_willing"]) if willing else float(penalties["abroad_no_relocate"])


# ══════════════════════════════════════════════════════════════
# PROTECTED SHORTLIST GATE
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# PROTECTED SHORTLIST GATE
# ══════════════════════════════════════════════════════════════

def compute_protected_score(candidate, cfg, skill_comp, career_comp):
    """
    Score used ONLY to rank candidates within the protected shortlist pool.
    Uses skill + career ONLY — deliberately excludes behavioral signals.

    Why no behavioral: the entire point of the protected shortlist is to
    rescue high-skill candidates who may have low behavioral scores (inactive,
    slow responder). Including behavioral here would mean a candidate with
    great skills but low engagement gets a low protected_score and falls out
    of the protected pool — defeating the purpose entirely.

    Weights: 60% skill, 40% career.
    Slightly skill-heavy: technical depth (skill) is harder to fake than
    company pedigree (career) and is the primary signal for this JD.

    This score is ONLY used to select the top-N from the protected pool.
    All final output is sorted by final_score (which includes behavioral).
    """
    return 0.60 * skill_comp + 0.40 * career_comp


def is_protected_eligible(candidate, cfg, desc_hits):
    """
    Tight gate: tier1_count >= 2 AND desc_hits >= 3.

    Gives ~370 eligible (~3.2% of passed pool) — meaningful safety net,
    not a backdoor for the entire pool.

    Previous gate (tier1_raw > 0 AND (ml_yoe > 0 OR desc_hits >= 3)) gave
    8,328 eligible (73% of pool) — useless as a bypass mechanism.

    Both conditions required (AND, not OR):
      tier1_count >= 2  — at least 2 distinct core retrieval/ML skills.
                          Rules out single-skill stuffers.
      desc_hits >= 3    — at least 3 retrieval/ML keywords in descriptions.
                          Ensures real work history, not just skill list gaming.

    A passing candidate has demonstrated depth (multiple tier1 skills) AND
    real work context (descriptions reference retrieval work).
    """
    tier1_set = {s.lower() for s in cfg["tier1_skills"]}
    aliases = cfg["skill_aliases"]

    # Unique tier1 skill names only — duplicate skill entries don't count twice
    cand_skill_names = {
        _normalize_skill_name(s.get("name", ""), aliases).lower()
        for s in candidate.get("skills", [])
    }
    tier1_count = len(cand_skill_names & tier1_set)

    if tier1_count < 2:
        return False
    if desc_hits < 3:
        return False

    return True


# ══════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION (for reasoning strings in Stage 6)
# ══════════════════════════════════════════════════════════════

def extract_signals(candidate, cfg):
    """
    Extracts key signals during scoring pass for use in reasoning generation.
    Stored on candidate dict as _signals.
    """
    import re
    aliases = cfg["skill_aliases"]
    tier1_set, _ = _get_tier_sets(cfg)
    raw_keywords = cfg["retrieval_keywords_raw"]

    desc_text = _collect_description_text(candidate)

    # Retrieval keywords found in descriptions (not skill list)
    desc_kw_found = [kw for kw in raw_keywords if kw in desc_text][:5]

    # Metrics/numbers from descriptions (regex: number + unit/context)
    metric_pattern = r'\d+(?:\.\d+)?(?:x|%|k|m|b|ms|qps|rpm|lpa)?\b'
    metrics_found = re.findall(metric_pattern, desc_text)[:3]

    # Role + company where retrieval keywords appeared
    retrieval_role = None
    for ch in reversed(candidate.get("career_history", [])):
        ch_desc = (ch.get("description") or "").lower()
        if any(kw in ch_desc for kw in raw_keywords):
            retrieval_role = f"{ch.get('title', '')} @ {ch.get('company', '')}"
            break

    # Top 2 trusted Tier1 skills
    tier1_skills_scored = []
    for s in candidate.get("skills", []):
        canonical = _normalize_skill_name(s.get("name", ""), aliases)
        if canonical.lower() in tier1_set:
            tier1_skills_scored.append((canonical, s.get("proficiency", ""), s.get("endorsements", 0)))
    tier1_skills_scored.sort(key=lambda x: (
        {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(x[1], 0),
        x[2]
    ), reverse=True)
    top_tier1 = tier1_skills_scored[:2]

    sig = candidate.get("redrob_signals", {}) or {}
    return {
        "desc_keywords": desc_kw_found,
        "metrics": metrics_found,
        "retrieval_role": retrieval_role,
        "top_tier1_skills": top_tier1,
        "response_rate": sig.get("recruiter_response_rate"),
        "notice_days": sig.get("notice_period_days"),
        "assessment_scores": sig.get("skill_assessment_scores") or {},
        "company_type": "product" if compute_product_ratio(candidate, cfg) > 0.3 else "services",
    }