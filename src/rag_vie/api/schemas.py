from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    index_dir: str | None = None
    mlp_path: str | None = None
    use_generator: bool = True


class RetrievedPassage(BaseModel):
    id: str
    passage: str
    score: float


class QueryResponse(BaseModel):
    query: str
    weights: dict[str, float]
    features: dict[str, float] = {}   # 8 Vietnamese linguistic features (router input)
    retrieved: list[RetrievedPassage]
    answer: str
    latency_ms: float
    index_dir: str
    mlp_path: str


class MethodResult(BaseModel):
    name: str
    label: str
    weights: dict[str, float]
    retrieved: list[RetrievedPassage]
    answer: str


class CompareResponse(BaseModel):
    query: str
    features: dict[str, float] = {}   # query-level, shared across all methods
    methods: list[MethodResult]
    latency_ms: float
    index_dir: str
    mlp_path: str


class IndexesResponse(BaseModel):
    index_dirs: list[str]
    mlp_checkpoints: list[str]
    default_index_dir: str
    default_mlp_path: str


class HealthResponse(BaseModel):
    status: str
