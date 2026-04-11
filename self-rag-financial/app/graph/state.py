from typing import TypedDict, List, Optional, Any

class SelfRAGState(TypedDict):
    # === INPUT ===
    query: str
    session_id: Optional[str]
    filters: Optional[dict]
    max_retries: Optional[int]

    # === ROUTING ===
    needs_retrieval: bool
    query_type: str

    # === RETRIEVAL ===
    active_query: str
    retrieved_chunks: List[dict]
    relevant_chunks: List[dict]
    retry_count: int

    # === GENERATION ===
    answer: str

    # === REFLECTION ===
    groundedness: str
    groundedness_confidence: float
    unsupported_claims: List[str]
    usefulness_score: int
    confidence: str

    # === METADATA ===
    sources: List[dict]
    response_time_ms: int
    cache_hit: bool
    failure_reason: Optional[str]
    disclaimer: Optional[str]
    suggested_filters: Optional[dict]
