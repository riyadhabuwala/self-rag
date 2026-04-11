import time
import logging
from functools import partial

from langgraph.graph import StateGraph, END
from app.graph.state import SelfRAGState
from app.graph.nodes import (
    router_node, guardrails_node, retrieve_node, grade_documents_node,
    rewrite_query_node, generate_node, hallucination_check_node,
    usefulness_check_node, finalize_node
)
from app.graph.edges import (
    route_after_router, route_after_grade_documents,
    route_after_hallucination_check, route_after_usefulness_check
)
from app.rag.graders import Graders
from app.rag.retriever import HybridRetriever
from app.rag.embedder import Embedder
from app.rag.chroma_store import ChromaStore
from app.rag.bm25_index import BM25Index

logger = logging.getLogger(__name__)

def build_graph(graders: Graders = None, retriever: HybridRetriever = None):
    if graders is None:
        graders = Graders()
    if retriever is None:
        embedder = Embedder()
        store = ChromaStore()
        bm25 = BM25Index()
        bm25.build(store)
        retriever = HybridRetriever(embedder, store, bm25)

    def bind(fn):
        return partial(fn, graders=graders, retriever=retriever)

    workflow = StateGraph(SelfRAGState)
    
    workflow.add_node("router", bind(router_node))
    workflow.add_node("guardrails", bind(guardrails_node))
    workflow.add_node("retrieve", bind(retrieve_node))
    workflow.add_node("grade_documents", bind(grade_documents_node))
    workflow.add_node("rewrite_query", bind(rewrite_query_node))
    workflow.add_node("generate", bind(generate_node))
    workflow.add_node("hallucination_check", bind(hallucination_check_node))
    workflow.add_node("usefulness_check", bind(usefulness_check_node))
    workflow.add_node("finalize", bind(finalize_node))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {"finalize": "finalize", "retrieve": "guardrails"}
    )
    workflow.add_edge("guardrails", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grade_documents,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "finalize": "finalize"
        }
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {
            "usefulness_check": "usefulness_check",
            "rewrite_query": "rewrite_query"
        }
    )
    workflow.add_conditional_edges(
        "usefulness_check",
        route_after_usefulness_check,
        {
            "finalize": "finalize",
            "rewrite_query": "rewrite_query"
        }
    )
    workflow.add_edge("finalize", END)

    graph = workflow.compile()
    logger.info("Self-RAG graph compiled successfully")
    return graph

def run_query(query: str, session_id: str = None, filters: dict = None, max_retries: int = None, graph=None) -> dict:
    if graph is None:
        graph = build_graph()

    initial_state = {
        "query": query,
        "session_id": session_id,
        "filters": filters,
        "max_retries": max_retries,
        "needs_retrieval": False,
        "query_type": "",
        "active_query": query,
        "retrieved_chunks": [],
        "relevant_chunks": [],
        "retry_count": 0,
        "answer": "",
        "groundedness": "",
        "groundedness_confidence": 0.0,
        "unsupported_claims": [],
        "usefulness_score": 0,
        "confidence": "",
        "sources": [],
        "response_time_ms": 0,
        "cache_hit": False,
        "failure_reason": None
    }

    start_time = time.time()
    final_state = graph.invoke(initial_state)
    elapsed_ms = int((time.time() - start_time) * 1000)

    final_state["response_time_ms"] = elapsed_ms
    return final_state
