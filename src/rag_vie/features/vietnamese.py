import re
import unicodedata

import numpy as np
from underthesea import word_tokenize

# Câu hỏi tiếng Việt
_QUESTION_WORDS = {
    "ai", "gì", "nào", "đâu", "khi nào", "tại sao", "vì sao",
    "như thế nào", "thế nào", "bao nhiêu", "bao lâu", "mấy",
}

# Từ kỹ thuật / chuyên ngành đơn giản (mở rộng bằng lexicon domain cụ thể sau)
_TECH_TERMS = {
    "cpu", "gpu", "ram", "server", "database", "api", "software",
    "hardware", "network", "internet", "website", "app", "algorithm",
}

# Dấu thanh điệu tiếng Việt
_TONED_PATTERN = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)

FEATURE_NAMES = [
    "diacritic_ratio",       # tỉ lệ âm tiết có dấu
    "compound_ratio",         # tỉ lệ từ ghép sau word segmentation
    "english_ratio",          # tỉ lệ token tiếng Anh (code-switching)
    "tech_term_ratio",        # tỉ lệ từ kỹ thuật/chuyên ngành
    "clause_count_norm",      # số mệnh đề (proxy cho multi-hop)
    "has_question_word",      # có từ để hỏi
    "query_length_norm",      # độ dài query (chuẩn hoá)
    "oov_ratio",              # tỉ lệ token không có trong BM25 vocab
]


def _has_diacritic(token: str) -> bool:
    return bool(_TONED_PATTERN.search(token))


def extract_features(query: str, bm25_vocab: set[str] | None = None) -> np.ndarray:
    """Extract Vietnamese-aware features from a query string.
    Returns a 1-D float32 array of length len(FEATURE_NAMES).
    bm25_vocab: set of tokens in the BM25 corpus vocabulary (from BM25Okapi.idf.keys()).
                When provided, enables OOV ratio calculation; otherwise oov_ratio=0.0.
    """
    syllables = query.strip().split()
    total_syl = max(len(syllables), 1)

    # 1. Diacritic ratio
    diacritic_ratio = sum(1 for s in syllables if _has_diacritic(s)) / total_syl

    # 2. Compound word ratio (từ ghép = word segment có >1 âm tiết, hiển thị là "_" join)
    segmented = word_tokenize(query, format="text")
    words = segmented.split()
    compound_ratio = sum(1 for w in words if "_" in w) / max(len(words), 1)

    # 3. English token ratio
    english_ratio = sum(1 for s in syllables if re.match(r"^[a-zA-Z]+$", s)) / total_syl

    # 4. Tech term ratio
    lower_syllables = [s.lower() for s in syllables]
    tech_term_ratio = sum(1 for s in lower_syllables if s in _TECH_TERMS) / total_syl

    # 5. Clause count (proxy for multi-hop complexity), normalized by 5
    clause_markers = re.findall(r"[,،]|\bvà\b|\bhoặc\b|\bnếu\b|\bnhưng\b|\btuy nhiên\b", query)
    clause_count_norm = min(len(clause_markers) / 5.0, 1.0)

    # 6. Question word presence
    query_lower = query.lower()
    has_question_word = float(any(qw in query_lower for qw in _QUESTION_WORDS))

    # 7. Query length, normalized at 20 syllables
    query_length_norm = min(total_syl / 20.0, 1.0)

    # 8. OOV ratio: fraction of BM25-tokenized query tokens absent from corpus vocab.
    #    Uses the same word_tokenize segmentation as BM25Retriever._tokenize().
    #    Returns 0.0 when bm25_vocab is not provided (e.g. at inference without index).
    if bm25_vocab is not None:
        oov_ratio = sum(1 for w in words if w not in bm25_vocab) / max(len(words), 1)
    else:
        oov_ratio = 0.0

    return np.array(
        [diacritic_ratio, compound_ratio, english_ratio, tech_term_ratio,
         clause_count_norm, has_question_word, query_length_norm, oov_ratio],
        dtype=np.float32,
    )
