"""Shared helpers for Stage 1 rule engine."""
from datetime import date, datetime


def parse_date(date_str):
    """Parse 'YYYY-MM-DD' string to date object. Returns None for null/invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def months_between(start, end):
    """Whole months between two date objects. end=None treated as today."""
    if start is None:
        return None
    if end is None:
        end = date.today()
    return (end.year - start.year) * 12 + (end.month - start.month)


def collect_candidate_text(candidate):
    """All free-text fields lowercased and joined — used for keyword presence checks.
    NOTE: career_history.description is intentionally EXCLUDED here — confirmed
    template-recycled across unrelated titles in the sample data, so it's an
    unreliable signal for keyword/skill-presence checks (see project findings).
    """
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
        candidate["profile"].get("current_title", ""),
    ]
    parts += [ch.get("title", "") for ch in candidate.get("career_history", [])]
    parts += [s.get("name", "") for s in candidate.get("skills", [])]
    return " ".join(parts).lower()


def all_companies(candidate):
    """Current company + every career_history company, for consulting-only check."""
    companies = [candidate["profile"].get("current_company", "")]
    companies += [ch.get("company", "") for ch in candidate.get("career_history", [])]
    return [c for c in companies if c]
