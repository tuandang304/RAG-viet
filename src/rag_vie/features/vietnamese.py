import re

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

# IDF cap dùng để chuẩn hoá đặc trưng độ hiếm về [0, 1].
# BM25Okapi idf ~ log((N - n + 0.5)/(n + 0.5)); với corpus ~10^5 docs giá trị max ~ log(N) ≈ 12.
_IDF_CAP = 12.0

FEATURE_NAMES = [
    "diacritic_ratio",       # tỉ lệ âm tiết có dấu
    "compound_ratio",         # tỉ lệ từ ghép sau word segmentation
    "english_ratio",          # tỉ lệ token tiếng Anh (code-switching)
    "tech_term_ratio",        # tỉ lệ từ kỹ thuật/chuyên ngành
    "clause_count_norm",      # số mệnh đề (proxy cho multi-hop)
    "has_question_word",      # có từ để hỏi
    "query_length_norm",      # độ dài query (chuẩn hoá)
    "oov_ratio",              # tỉ lệ token không có trong BM25 vocab
    "avg_idf_norm",           # độ hiếm trung bình của token (IDF) → giá trị của lexical/sparse
    "max_idf_norm",           # token hiếm nhất (một thuật ngữ rất hiếm → exact-match quan trọng)
    "digit_ratio",            # tỉ lệ âm tiết chứa chữ số (số liệu/ngày tháng → exact-match)
    "proper_noun_ratio",      # tỉ lệ âm tiết viết hoa giữa câu (thực thể → dense/exact)
    "acronym_ratio",          # tỉ lệ token viết tắt (IN HOA, ví dụ "WTO", "GDP")
]


def _has_diacritic(token: str) -> bool:
    return bool(_TONED_PATTERN.search(token))


def _norm_idf(idf: float) -> float:
    """Clip BM25 idf về [0, _IDF_CAP] rồi chuẩn hoá [0, 1] (idf âm cho token rất phổ biến → 0)."""
    return min(max(idf, 0.0), _IDF_CAP) / _IDF_CAP


def extract_features(query: str, bm25_idf: dict[str, float] | None = None) -> np.ndarray:
    """Extract Vietnamese-aware features from a query string.

    Returns a 1-D float32 array of length ``len(FEATURE_NAMES)``.

    ``bm25_idf``: mapping token → BM25 IDF từ ``BM25Okapi.idf`` của corpus. Khi được
    cung cấp sẽ bật các đặc trưng phụ thuộc corpus (``oov_ratio``, ``avg_idf_norm``,
    ``max_idf_norm``); nếu ``None`` các đặc trưng này = 0 (suy luận khi chưa có index).
    """
    syllables = query.strip().split()
    total_syl = max(len(syllables), 1)

    # 1. Diacritic ratio
    diacritic_ratio = sum(1 for s in syllables if _has_diacritic(s)) / total_syl

    # 2. Compound word ratio (từ ghép = word segment có >1 âm tiết, hiển thị là "_" join)
    segmented = word_tokenize(query, format="text")
    words = segmented.split()
    n_words = max(len(words), 1)
    compound_ratio = sum(1 for w in words if "_" in w) / n_words

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

    # 8–10. BM25 corpus-dependent features (OOV + IDF specificity).
    #       Dùng cùng segmentation với BM25Retriever._tokenize() để khớp vocab.
    if bm25_idf is not None:
        oov_ratio = sum(1 for w in words if w not in bm25_idf) / n_words
        idf_vals = [_norm_idf(bm25_idf[w]) for w in words if w in bm25_idf]
        avg_idf_norm = float(np.mean(idf_vals)) if idf_vals else 0.0
        max_idf_norm = float(np.max(idf_vals)) if idf_vals else 0.0
    else:
        oov_ratio = 0.0
        avg_idf_norm = 0.0
        max_idf_norm = 0.0

    # 11. Digit ratio (số liệu, năm, mã số → lexical/exact-match)
    digit_ratio = sum(1 for s in syllables if any(ch.isdigit() for ch in s)) / total_syl

    # 12. Proper-noun ratio: âm tiết viết hoa KHÔNG phải token đầu câu (thực thể tiếng Việt).
    proper_noun_ratio = (
        sum(1 for s in syllables[1:] if s[:1].isupper() and s[1:].islower()) / total_syl
    )

    # 13. Acronym ratio: token IN HOA dài ≥2 ký tự chữ (WTO, GDP, USD…).
    acronym_ratio = sum(
        1 for s in syllables if len(s) >= 2 and s.isupper() and s.isalpha()
    ) / total_syl

    return np.array(
        [diacritic_ratio, compound_ratio, english_ratio, tech_term_ratio,
         clause_count_norm, has_question_word, query_length_norm, oov_ratio,
         avg_idf_norm, max_idf_norm, digit_ratio, proper_noun_ratio, acronym_ratio],
        dtype=np.float32,
    )
