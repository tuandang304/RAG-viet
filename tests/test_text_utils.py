from rag_vie.utils.text import remove_diacritics


def test_remove_diacritics_basic():
    assert remove_diacritics("bệnh tiểu đường") == "benh tieu duong"


def test_remove_diacritics_handles_d_bar():
    assert remove_diacritics("Đà Nẵng đẹp") == "Da Nang dep"


def test_remove_diacritics_ascii_passthrough():
    assert remove_diacritics("CPU server 2024") == "CPU server 2024"
