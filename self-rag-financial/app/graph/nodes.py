import logging
from typing import List, Dict, Any

from app.graph.state import SelfRAGState
from app.rag.graders import Graders
from app.rag.retriever import HybridRetriever
from app.rag.retriever_utils import compute_confidence_from_scores
from app.config import settings
from app.rag.extractor import FinancialExtractor

logger = logging.getLogger(__name__)

extractor = FinancialExtractor()

def guardrails_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        query = state.get("query", "")
        res = extractor.classify_query_intent(query)
        logger.info(f"[GUARDRAILS] intent={res.get('intent')}, requires_disclaimer={res.get('requires_disclaimer')}")
        
        update = {}
        if res.get("requires_disclaimer"):
            disclaimer = (
                "DISCLAIMER: This system provides factual information from financial "
                "documents only. Nothing in this response constitutes investment advice, "
                "a recommendation to buy or sell securities, or a prediction of future "
                "performance. Consult a licensed financial advisor for investment decisions."
            )
            update["disclaimer"] = disclaimer
            update["needs_retrieval"] = True

        ext_tickers = res.get("extracted_tickers", [])
        current_filters = state.get("filters")
        
        if ext_tickers and current_filters is None:
            update["suggested_filters"] = {"ticker": ext_tickers[0]}
            
        return update
    except Exception as e:
        logger.error(f"[GUARDRAILS] Error: {e}")
        return {}

def router_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        res = graders.grade_retrieval_needed(state["query"])
        logger.info(f"[ROUTER] query_type={res.get('query_type')}, needs_retrieval={res.get('needs_retrieval')}")
        return {
            "needs_retrieval": res.get("needs_retrieval", True),
            "query_type": res.get("query_type", "financial_factual"),
            "active_query": state["query"],
            "retry_count": 0,
            "cache_hit": False
        }
    except Exception as e:
        logger.error(f"[ROUTER] Error: {e}")
        return {
            "needs_retrieval": True,
            "query_type": "financial_factual",
            "active_query": state["query"],
            "retry_count": 0,
            "cache_hit": False
        }

def retrieve_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        chunks = retriever.retrieve(
            query=state.get("active_query", state["query"]),
            filters=state.get("filters"),
            use_multi_query=True
        )
        logger.info(f"[RETRIEVE] query='{state.get('active_query', state['query'])}' -> {len(chunks)} chunks")
        return {"retrieved_chunks": chunks}
    except Exception as e:
        logger.error(f"[RETRIEVE] Error: {e}")
        return {"retrieved_chunks": [], "failure_reason": f"Retrieval failed: {e}"}

def grade_documents_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        chunks = state.get("retrieved_chunks", [])
        chunks_with_verdicts = []
        relevant_chunks = []
        for chunk in chunks:
            res = graders.grade_document_relevance(query=state["query"], chunk_text=chunk.get("text", ""))
            chunk_copy = dict(chunk)
            chunk_copy["relevance_verdict"] = res
            chunks_with_verdicts.append(chunk_copy)
            if res.get("verdict") == "relevant":
                relevant_chunks.append(chunk_copy)
                
        logger.info(f"[GRADE_DOCS] {len(relevant_chunks)}/{len(chunks)} chunks relevant")
        
        failure_reason = "No relevant documents found for this query" if len(relevant_chunks) == 0 else None
        return {
            "retrieved_chunks": chunks_with_verdicts,
            "relevant_chunks": relevant_chunks,
            "failure_reason": failure_reason
        }
    except Exception as e:
        logger.error(f"[GRADE_DOCS] Error: {e}")
        return {
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "relevant_chunks": [],
            "failure_reason": "Grading documents failed"
        }

def rewrite_query_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        retry_count = state.get("retry_count", 0)
        res = graders.rewrite_query(
            query=state.get("active_query", state["query"]),
            failure_reason=state.get("failure_reason", "No relevant documents found"),
            attempt_number=retry_count + 1
        )
        new_query = res.get("rewritten_query", state.get("active_query", state["query"]))
        logger.info(f"[REWRITE] attempt {retry_count+1} -> '{new_query}'")
        return {
            "active_query": new_query,
            "retry_count": retry_count + 1,
            "retrieved_chunks": [],
            "relevant_chunks": []
        }
    except Exception as e:
        logger.error(f"[REWRITE] Error: {e}")
        return {
            "active_query": state.get("active_query", state["query"]),
            "retry_count": state.get("retry_count", 0) + 1,
            "retrieved_chunks": [],
            "relevant_chunks": []
        }

def generate_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        relevant = state.get("relevant_chunks", [])
        answer = graders.generate_answer(query=state["query"], context_chunks=relevant)
        
        sources = []
        seen = set()
        for chunk in relevant:
            meta = chunk.get("metadata", {})
            doc = meta.get("source_file", "")
            page = meta.get("page_number")
            key = f"{doc}::{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "document": doc,
                    "page": page,
                    "ticker": meta.get("ticker"),
                    "fiscal_year": meta.get("fiscal_year"),
                    "chunk_id": chunk.get("chunk_id")
                })
                
        logger.info(f"[GENERATE] answer length={len(answer)} chars, sources={len(sources)}")
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"[GENERATE] Error: {e}")
        return {"answer": "Error generating answer.", "sources": []}

def hallucination_check_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        context_parts = []
        for chunk in state.get("relevant_chunks", []):
            meta = chunk.get('metadata', {})
            src = meta.get('source_file', 'unknown')
            page = meta.get('page_number', '?')
            ticker = meta.get('ticker', '?')
            yr = meta.get('fiscal_year', '?')
            text = chunk.get('text', '')
            context_parts.append(f"[Source: {src}, Page {page}, Ticker: {ticker}, Year: {yr}]\n{text}")
            
        context_string = "\n\n---\n\n".join(context_parts)
        
        res = graders.check_hallucination(
            query=state["query"],
            context=context_string,
            answer=state.get("answer", "")
        )
        logger.info(f"[HALLUCINATION] verdict={res.get('verdict')}, confidence={res.get('confidence', 0):.2f}, unsupported_claims={len(res.get('unsupported_claims', []))}")
        return {
            "groundedness": res.get("verdict", "no"),
            "groundedness_confidence": res.get("confidence", 0.0),
            "unsupported_claims": res.get("unsupported_claims", [])
        }
    except Exception as e:
        logger.error(f"[HALLUCINATION] Error: {e}")
        return {
            "groundedness": "no",
            "groundedness_confidence": 0.0,
            "unsupported_claims": []
        }

def usefulness_check_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    try:
        res = graders.grade_usefulness(
            query=state["query"],
            answer=state.get("answer", "")
        )
        score = res.get("score", 1)
        conf = compute_confidence_from_scores(
            groundedness=state.get("groundedness", "no"),
            usefulness_score=score
        )
        logger.info(f"[USEFULNESS] score={score}/5, confidence={conf}")
        return {
            "usefulness_score": score,
            "confidence": conf
        }
    except Exception as e:
        logger.error(f"[USEFULNESS] Error: {e}")
        return {"usefulness_score": 1, "confidence": "low"}

def finalize_node(state: SelfRAGState, graders: Graders, retriever: HybridRetriever) -> dict:
    update = {}
    try:
        if not state.get("needs_retrieval", True):
            answer = graders.generate_answer(query=state["query"], context_chunks=[])
            update["answer"] = answer
            update["groundedness"] = "fully"
            update["confidence"] = "high"
            update["usefulness_score"] = 5
        elif state.get("failure_reason") and len(state.get("relevant_chunks", [])) == 0:
            query = state.get("query", "")
            answer = (f"I was unable to find relevant information in the "
                      f"indexed documents to answer your question: '{query}'. "
                      "Please ensure the relevant documents have been ingested, or try rephrasing your query.")
            update["answer"] = answer
            update["groundedness"] = "no"
            update["confidence"] = "low"
            update["usefulness_score"] = 1
            
        conf = update.get("confidence", state.get("confidence", "N/A"))
        ground = update.get("groundedness", state.get("groundedness", "N/A"))
        logger.info(f"[FINALIZE] confidence={conf}, groundedness={ground}")
        return update
    except Exception as e:
        logger.error(f"[FINALIZE] Error: {e}")
        return update
