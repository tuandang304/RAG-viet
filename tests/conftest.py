"""Shared test setup.

Makes the suite runnable without a real .env (e.g. on CI): rag_vie.config
instantiates Settings at import time, so required FPT fields must exist as
environment variables before any rag_vie import happens.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("FPT_API_KEY", "test-key")
os.environ.setdefault("FPT_BASE_URL", "http://localhost:9999/v1")
