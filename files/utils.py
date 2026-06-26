"""
Shared helpers for Stage 1 rule engine.
All recency calculations anchor to REFERENCE_DATE.
"""
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 26)

# ── Industries that are pure services/consulting (not product) ──────────────
SERVICES_INDUSTRIES = {"IT Services", "Consulting", "Staffing", "Outsourcing", "BPO/KPO"}

# Industries that exist in the dataset but are NOT product-tech companies
# (used in F6 — zero product experience)
NON_PRODUCT_INDUSTRIES = SERVICES_INDUSTRIES | {"Manufacturing", "Conglomerate", "Paper Products", ""}

# ── JD-relevant skill names (exact match against skills[].name) ─────────────
TIER1_SKILLS = {
    # Core retrieval / vector DB
    "Embeddings", "FAISS", "Pinecone", "Elasticsearch", "OpenSearch",
    "Weaviate", "Qdrant", "Milvus", "Vector Search", "Information Retrieval",
    "BM25", "Sentence Transformers",
    # ML / NLP
    "Recommendation Systems", "RAG", "LangChain", "Hugging Face Transformers",
    "Fine-tuning LLMs", "Haystack",
    # Supporting ML
    "Feature Engineering", "scikit-learn", "PyTorch", "MLflow",
}

# ── Keywords that must appear in career descriptions for retrieval signal ────
# Used in F5 (non-tech title check) and Stage 4 description scoring
RETRIEVAL_DESC_KEYWORDS = {
    "embedding", "vector", "retrieval", "faiss", "pinecone", "elasticsearch",
    "opensearch", "recommendation", "ranking", "rerank", "re-rank",
    "semantic search", "dense retrieval", "information retrieval", "bm25",
    "sentence transformer", "learning to rank", "neural search",
    "approximate nearest", "ann index", "rag", "langchain",
    "hugging face", "fine-tun", "feature engineering", "pytorch",
    "weaviate", "qdrant", "milvus", "vector db", "vector store",
    "similarity search", "knn search", "nearest neighbour",
}

# ── Non-technical title substrings (F5 disqualifier when + zero retrieval) ──
NON_TECH_TITLE_KEYWORDS = {
    "business analyst", "hr manager", "human resources", "mechanical engineer",
    "accountant", "project manager", "customer support", "operations manager",
    "content writer", "sales executive", "civil engineer", "graphic designer",
    "marketing manager", "financial analyst", "supply chain", "logistics",
    "legal counsel", "recruiter", "tax consultant", "quality analyst",
    "social media", "procurement", "administrative", "account manager",
    "sales manager", "brand manager", "product designer", "ux designer",
    "ui designer", "interior design", "fashion", "teacher", "professor",
}


# ── Date helpers ─────────────────────────────────────────────────────────────

def parse_date(date_str: str | None) -> date | None:
    """Parse 'YYYY-MM-DD' string → date object. Returns None for null/invalid."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def months_between(start: date | None, end: date | None = None) -> int | None:
    """Whole months between two date objects. end=None → REFERENCE_DATE."""
    if start is None:
        return None
    if end is None:
        end = REFERENCE_DATE
    return (end.year - start.year) * 12 + (end.month - start.month)


def days_since(date_str: str | None) -> int:
    """Days between date_str and REFERENCE_DATE. Returns 9999 if unparseable."""
    d = parse_date(date_str)
    if d is None:
        return 9999
    return (REFERENCE_DATE - d).days


# ── Candidate field collectors ───────────────────────────────────────────────

def all_companies(candidate: dict) -> list[str]:
    """All company names from profile + entire career history."""
    companies = [candidate["profile"].get("current_company", "")]
    companies += [ch.get("company", "") for ch in candidate.get("career_history", [])]
    return [c for c in companies if c]


def all_ch_industries(candidate: dict) -> list[str]:
    """All industry strings from career_history entries."""
    return [ch.get("industry", "") for ch in candidate.get("career_history", [])]


def collect_description_text(candidate: dict) -> str:
    """
    All career_history descriptions concatenated — lowercased.
    Used ONLY for retrieval keyword presence check (F5 + Stage 4).
    Kept separate from skill/title text to avoid cross-contamination.
    """
    parts = [ch.get("description", "") for ch in candidate.get("career_history", [])]
    return " ".join(p for p in parts if p).lower()


def collect_profile_text(candidate: dict) -> str:
    """
    Headline + summary + titles + skill names — lowercased.
    Does NOT include career descriptions (template-recycled risk).
    """
    profile = candidate["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    parts += [ch.get("title", "") for ch in candidate.get("career_history", [])]
    parts += [s.get("name", "") for s in candidate.get("skills", [])]
    return " ".join(p for p in parts if p).lower()


def has_retrieval_in_descriptions(candidate: dict) -> bool:
    """True if any RETRIEVAL_DESC_KEYWORDS appear in career descriptions."""
    desc = collect_description_text(candidate)
    return any(kw in desc for kw in RETRIEVAL_DESC_KEYWORDS)


def has_tier1_skill(candidate: dict) -> bool:
    """True if candidate has at least one TIER1_SKILLS match."""
    cand_skills = {s.get("name", "") for s in candidate.get("skills", [])}
    return bool(cand_skills & TIER1_SKILLS)


def is_non_tech_title(candidate: dict) -> bool:
    """True if current_title matches any known non-technical title keyword."""
    title = candidate["profile"].get("current_title", "").lower()
    return any(nt in title for nt in NON_TECH_TITLE_KEYWORDS)


def has_product_experience(candidate: dict) -> bool:
    """True if at least one career_history role is at a product-tech company."""
    industries = all_ch_industries(candidate)
    return any(i not in NON_PRODUCT_INDUSTRIES for i in industries)
