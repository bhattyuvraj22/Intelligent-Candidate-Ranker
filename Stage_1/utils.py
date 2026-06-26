"""
Shared helpers for Stage 1 rule engine.
All recency calculations anchor to REFERENCE_DATE.
"""
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 26)

# ── Industries ───────────────────────────────────────────────────────────────
SERVICES_INDUSTRIES = {"IT Services", "Consulting", "Staffing", "Outsourcing", "BPO/KPO"}
NON_PRODUCT_INDUSTRIES = SERVICES_INDUSTRIES | {"Manufacturing", "Conglomerate", "Paper Products", ""}

# ── Tier 1 skills — core retrieval / vector DB / ML (exact name match) ──────
TIER1_SKILLS = {
    "Embeddings", "FAISS", "Pinecone", "Elasticsearch", "OpenSearch",
    "Weaviate", "Qdrant", "Milvus", "Vector Search", "Information Retrieval",
    "BM25", "Sentence Transformers", "Recommendation Systems", "RAG",
    "LangChain", "Hugging Face Transformers", "Fine-tuning LLMs", "Haystack",
    "Feature Engineering", "scikit-learn", "PyTorch", "MLflow",
    "Learning to Rank",   # added v4 — was in analysis set but missing from code
}

# ── Tier 2 skills — strong ML/AI adjacent (used in tier2 rescue pass) ───────
TIER2_SKILLS = {
    "TensorFlow", "Deep Learning", "NLP", "MLOps", "Data Science",
    "Computer Vision", "Keras", "BERT", "Transformers", "LLM",
    "Text Mining", "Named Entity Recognition", "Reinforcement Learning",
    "Transfer Learning", "Model Deployment", "A/B Testing",
    "Experiment Tracking", "Model Serving", "Semantic Search",
}

# ── AI/ML title keywords — used for tier2 rescue condition ──────────────────
AI_TITLE_KEYWORDS = {
    "ml engineer", "machine learning", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist", "deep learning",
    "ai research", "computer vision engineer", "recommendation", "search engineer",
    "applied ml", "senior ml", "junior ml", "ai/ml", "llm engineer",
    "data engineer",
}

# ── Retrieval keywords for description check ─────────────────────────────────
RETRIEVAL_DESC_KEYWORDS = {
    "embedding", "vector", "retrieval", "faiss", "pinecone", "elasticsearch",
    "opensearch", "recommendation", "ranking", "rerank", "re-rank",
    "semantic search", "dense retrieval", "information retrieval", "bm25",
    "sentence transformer", "learning to rank", "neural search",
    "approximate nearest", "ann index", "rag", "langchain",
    "hugging face", "fine-tun", "feature engineering", "pytorch",
    "weaviate", "qdrant", "milvus", "vector db", "vector store",
    "similarity search", "knn search", "nearest neighbour",
    "collaborative filtering", "matrix factorization", "re-ranking",
    "gradient boosted ranking", "neural ranking", "sparse retrieval",
}

# ── Non-technical titles — F5 disqualifier when + zero retrieval ─────────────
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

# ── Expected endorsements/duration per proficiency (from dataset medians) ────
PROFICIENCY_EXPECTED = {
    "beginner":     {"endorsements": 7,  "duration_months": 10},
    "intermediate": {"endorsements": 7,  "duration_months": 21},
    "advanced":     {"endorsements": 16, "duration_months": 29},
    "expert":       {"endorsements": 28, "duration_months": 65},
}


# ── Date helpers ─────────────────────────────────────────────────────────────

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def months_between(start, end=None):
    if start is None:
        return None
    if end is None:
        end = REFERENCE_DATE
    return (end.year - start.year) * 12 + (end.month - start.month)


def days_since(date_str):
    d = parse_date(date_str)
    if d is None:
        return 9999
    return (REFERENCE_DATE - d).days


# ── Candidate field helpers ──────────────────────────────────────────────────

def all_companies(candidate):
    companies = [candidate["profile"].get("current_company", "")]
    companies += [ch.get("company", "") for ch in candidate.get("career_history", [])]
    return [c for c in companies if c]


def all_ch_industries(candidate):
    return [ch.get("industry", "") for ch in candidate.get("career_history", [])]


def collect_description_text(candidate):
    """All career descriptions concatenated — lowercased. Used for keyword checks only."""
    parts = [ch.get("description", "") for ch in candidate.get("career_history", [])]
    return " ".join(p for p in parts if p).lower()


def collect_profile_text(candidate):
    """Headline + summary + titles + skill names — lowercased. No descriptions."""
    profile = candidate["profile"]
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    parts += [ch.get("title", "") for ch in candidate.get("career_history", [])]
    parts += [s.get("name", "") for s in candidate.get("skills", [])]
    return " ".join(p for p in parts if p).lower()


def has_retrieval_in_descriptions(candidate):
    desc = collect_description_text(candidate)
    return any(kw in desc for kw in RETRIEVAL_DESC_KEYWORDS)


def has_tier1_skill(candidate):
    cand_skills = {s.get("name", "") for s in candidate.get("skills", [])}
    return bool(cand_skills & TIER1_SKILLS)


def has_tier2_rescue(candidate):
    """
    Tier2 rescue: AI title + product exp + right YoE + 2+ tier2 skills
    + retrieval in descriptions.
    Validated: rescues exactly CAND_0006833 in dataset.
    """
    yoe = candidate["profile"].get("years_of_experience") or 0
    if yoe < 3 or yoe > 15:
        return False
    title = candidate["profile"].get("current_title", "").lower()
    if not any(kw in title for kw in AI_TITLE_KEYWORDS):
        return False
    industries = all_ch_industries(candidate)
    if not any(i not in NON_PRODUCT_INDUSTRIES for i in industries):
        return False
    cand_skills = {s.get("name", "") for s in candidate.get("skills", [])}
    if len(cand_skills & TIER2_SKILLS) < 2:
        return False
    return has_retrieval_in_descriptions(candidate)


def is_non_tech_title(candidate):
    title = candidate["profile"].get("current_title", "").lower()
    return any(nt in title for nt in NON_TECH_TITLE_KEYWORDS)


def has_product_experience(candidate):
    industries = all_ch_industries(candidate)
    return any(i not in NON_PRODUCT_INDUSTRIES for i in industries)


def get_earliest_graduation_year(candidate):
    """Returns earliest education end_year. Schema uses integer years, not date strings."""
    years = [e.get("end_year") for e in candidate.get("education", []) if e.get("end_year")]
    return min(years) if years else None


def get_earliest_job_start_year(candidate):
    """Returns earliest career role start year parsed from start_date string."""
    years = []
    for ch in candidate.get("career_history", []):
        sd = ch.get("start_date", "")
        if sd and len(sd) >= 4:
            try:
                years.append(int(sd[:4]))
            except ValueError:
                pass
    return min(years) if years else None