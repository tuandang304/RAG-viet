import unicodedata


def remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritical marks from text.

    Simulates users typing without tone marks, e.g.:
    "bệnh tiểu đường" → "benh tieu duong"
    """
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")
