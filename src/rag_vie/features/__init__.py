# Lazy package — import submodules trực tiếp (vd: `from rag_vie.features.signal import ...`).
# Không eager-import vietnamese ở đây vì nó kéo theo underthesea/torch (nặng, và không cần
# cho các module thuần như features.signal).
