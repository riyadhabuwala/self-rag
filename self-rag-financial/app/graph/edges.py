from app.graph.state import SelfRAGState
from app.config import settings

def route_after_router(state: SelfRAGState) -> str:
    if state.get("needs_retrieval", True) is False:
        return "finalize"
    return "retrieve"

def route_after_grade_documents(state: SelfRAGState) -> str:
    max_retries = state.get("max_retries")
    if max_retries is None:
        max_retries = settings.MAX_RETRIES
        
    if len(state.get("relevant_chunks", [])) > 0:
        return "generate"
    if state.get("retry_count", 0) >= max_retries:
        return "finalize"
    return "rewrite_query"

def route_after_hallucination_check(state: SelfRAGState) -> str:
    max_retries = state.get("max_retries")
    if max_retries is None:
        max_retries = settings.MAX_RETRIES
        
    groundedness = state.get("groundedness", "no")
    if groundedness == "fully":
        return "usefulness_check"
    if groundedness == "partially":
        return "usefulness_check"
    if groundedness == "no":
        if state.get("retry_count", 0) >= max_retries:
            return "usefulness_check"
        return "rewrite_query"
        
    return "rewrite_query"

def route_after_usefulness_check(state: SelfRAGState) -> str:
    max_retries = state.get("max_retries")
    if max_retries is None:
        max_retries = settings.MAX_RETRIES
        
    score = state.get("usefulness_score", 1)
    if score >= settings.USEFULNESS_THRESHOLD:
        return "finalize"
    if state.get("retry_count", 0) >= max_retries:
        return "finalize"
    return "rewrite_query"
