"""LLM Graders for Self-RAG System"""
import json
import logging
import re
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from app.config import settings
from app.rag.prompts import (
    RETRIEVAL_ROUTER_PROMPT, RETRIEVAL_ROUTER_USER,
    DOCUMENT_RELEVANCE_PROMPT, DOCUMENT_RELEVANCE_USER,
    HALLUCINATION_CHECK_PROMPT, HALLUCINATION_CHECK_USER,
    USEFULNESS_PROMPT, USEFULNESS_USER,
    QUERY_REWRITE_PROMPT, QUERY_REWRITE_USER
)

logger = logging.getLogger(__name__)

class Graders:
    def __init__(self):
        self.grading_llm = ChatGroq(
            model=settings.GROQ_GRADING_MODEL,
            temperature=0.0,
            max_tokens=settings.GROQ_MAX_TOKENS,
            api_key=settings.GROQ_API_KEY
        )
        
        self.generation_llm = ChatGroq(
            model=settings.GROQ_GENERATION_MODEL,
            temperature=0.1,
            max_tokens=settings.GROQ_MAX_TOKENS,
            api_key=settings.GROQ_API_KEY
        )

    def _call_grader(self, system_prompt: str, user_message: str, use_generation_llm: bool = False, expect_json: bool = True) -> dict:
        llm = self.generation_llm if use_generation_llm else self.grading_llm
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]
        
        response = llm.invoke(messages)
        text = response.content.strip()
        
        if not expect_json:
            return {"text": text}
            
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        return json.loads(text)

    def grade_retrieval_needed(self, query: str) -> dict:
        fallback = {
            "needs_retrieval": True,
            "reason": "Grader error — defaulting to retrieval",
            "query_type": "financial_factual"
        }
        try:
            user_msg = RETRIEVAL_ROUTER_USER.format(query=query)
            res = self._call_grader(RETRIEVAL_ROUTER_PROMPT, user_msg, expect_json=True)
            
            if not isinstance(res, dict):
                return fallback
                
            if "needs_retrieval" not in res:
                res["needs_retrieval"] = True
            if "reason" not in res:
                res["reason"] = "Valid response but missing reason"
            if "query_type" not in res:
                res["query_type"] = "financial_factual"
                
            res["needs_retrieval"] = bool(res["needs_retrieval"])
            return res
        except Exception as e:
            logger.warning(f"Grader failed: {e}")
            return fallback

    def grade_document_relevance(self, query: str, chunk_text: str) -> dict:
        fallback = {
            "verdict": "relevant",
            "reason": "Grader error — defaulting to relevant",
            "relevance_score": 0.5
        }
        try:
            user_msg = DOCUMENT_RELEVANCE_USER.format(query=query, chunk_text=chunk_text)
            res = self._call_grader(DOCUMENT_RELEVANCE_PROMPT, user_msg, expect_json=True)
            
            if not isinstance(res, dict):
                return fallback
                
            verdict = str(res.get("verdict", "relevant")).lower()
            if verdict not in ["relevant", "irrelevant"]:
                verdict = "relevant"
                
            try:
                score = float(res.get("relevance_score", 0.5))
            except (ValueError, TypeError):
                score = 0.5
                
            score = max(0.0, min(1.0, score))
            
            return {
                "verdict": verdict,
                "reason": str(res.get("reason", "No reason provided")),
                "relevance_score": score
            }
        except Exception as e:
            logger.warning(f"Grader failed: {e}")
            return fallback

    def check_hallucination(self, query: str, context: str, answer: str) -> dict:
        fallback = {
            "verdict": "partially",
            "confidence": 0.5,
            "unsupported_claims": [],
            "reason": "Grader error — could not verify grounding"
        }
        try:
            user_msg = HALLUCINATION_CHECK_USER.format(query=query, context=context, answer=answer)
            res = self._call_grader(HALLUCINATION_CHECK_PROMPT, user_msg, expect_json=True)
            
            if not isinstance(res, dict):
                return fallback
                
            verdict = str(res.get("verdict", "partially")).lower()
            if verdict not in ["fully", "partially", "no"]:
                verdict = "partially"
                
            try:
                confidence = float(res.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
                
            confidence = max(0.0, min(1.0, confidence))
            
            unsupported_claims = res.get("unsupported_claims", [])
            if not isinstance(unsupported_claims, list):
                unsupported_claims = []
                
            return {
                "verdict": verdict,
                "confidence": confidence,
                "unsupported_claims": unsupported_claims,
                "reason": str(res.get("reason", "No reason provided"))
            }
        except Exception as e:
            logger.warning(f"Grader failed: {e}")
            return fallback

    def grade_usefulness(self, query: str, answer: str) -> dict:
        fallback = {
            "score": 3,
            "reason": "Grader error — defaulting to neutral score"
        }
        try:
            user_msg = USEFULNESS_USER.format(query=query, answer=answer)
            res = self._call_grader(USEFULNESS_PROMPT, user_msg, expect_json=True)
            
            if not isinstance(res, dict):
                return fallback
                
            try:
                score = int(res.get("score", 3))
            except (ValueError, TypeError):
                score = 3
                
            score = max(1, min(5, score))
            
            return {
                "score": score,
                "reason": str(res.get("reason", "No reason provided"))
            }
        except Exception as e:
            logger.warning(f"Grader failed: {e}")
            return fallback

    def rewrite_query(self, query: str, failure_reason: str, attempt_number: int) -> dict:
        fallback = {
            "rewritten_query": query + " financial statements SEC filing",
            "strategy": "Grader error — applied default financial suffix",
            "key_terms_added": ["financial statements", "SEC filing"]
        }
        try:
            user_msg = QUERY_REWRITE_USER.format(
                query=query,
                failure_reason=failure_reason,
                attempt_number=attempt_number,
                max_attempts=settings.MAX_RETRIES
            )
            res = self._call_grader(QUERY_REWRITE_PROMPT, user_msg, expect_json=True)
            
            if not isinstance(res, dict):
                return fallback
                
            rewritten_query = str(res.get("rewritten_query", ""))
            if not rewritten_query:
                rewritten_query = fallback["rewritten_query"]
                
            key_terms = res.get("key_terms_added", [])
            if not isinstance(key_terms, list):
                key_terms = []
                
            return {
                "rewritten_query": rewritten_query,
                "strategy": str(res.get("strategy", fallback["strategy"])),
                "key_terms_added": key_terms
            }
        except Exception as e:
            logger.warning(f"Grader failed: {e}")
            return fallback

    def generate_answer(self, query: str, context_chunks: List[dict]) -> str:
        fallback = "I was unable to generate an answer due to a technical error. Please try again."
        try:
            context_parts = []
            for chunk in context_chunks:
                meta = chunk.get('metadata', {})
                src = meta.get('source_file', 'unknown')
                page = meta.get('page_number', '?')
                ticker = meta.get('ticker', '?')
                yr = meta.get('fiscal_year', '?')
                text = chunk.get('text', '')
                
                context_parts.append(
                    f"[Source: {src}, Page {page}, Ticker: {ticker}, Year: {yr}]\\n{text}"
                )
                
            context_string = "\\n\\n---\\n\\n".join(context_parts)
            
            system_prompt = """You are a financial document analyst. Answer the user's question using ONLY the provided source context. Follow these rules strictly:
1. Cite your sources inline using [Source: filename, Page X] format
2. Use only facts present in the provided context
3. If the context does not contain enough information, say explicitly: "The provided documents do not contain sufficient information to answer this question."
4. Do not add general financial knowledge not present in the context
5. If documents contradict each other, note the contradiction explicitly
6. Keep the answer concise and structured — use bullet points for lists of figures"""

            user_msg = f"Context:\\n{context_string}\\n\\nQuestion: {query}\\n\\nAnswer:"
            
            res = self._call_grader(system_prompt, user_msg, use_generation_llm=True, expect_json=False)
            return str(res.get("text", fallback))
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return fallback