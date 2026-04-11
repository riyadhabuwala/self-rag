"""Retriever Utilities"""
from typing import List

def build_chroma_filter(filters: dict) -> dict | None:
    if not filters:
        return None
    
    valid = []
    for k, v in filters.items():
        if v is not None and v != "":
            valid.append({k: {"$eq": v}})
            
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    return {"$and": valid}

def deduplicate_by_chunk_id(results: List[dict], keep: str = "best_rank") -> List[dict]:
    seen = {}
    for r in results:
        cid = r["chunk_id"]
        if cid not in seen:
            seen[cid] = r
        else:
            if keep == "best_rank":
                # Assuming lower rank is better
                current_rank = seen[cid].get("rank", float('inf'))
                new_rank = r.get("rank", float('inf'))
                if new_rank < current_rank:
                    seen[cid] = r
    return list(seen.values())

def compute_confidence_from_scores(groundedness: str, usefulness_score: int) -> str:
    # "high": groundedness="fully" AND usefulness_score >= 4
    if groundedness == "fully" and usefulness_score >= 4:
        return "high"
    # "low": groundedness="no" OR usefulness_score <= 2
    if groundedness == "no" or usefulness_score <= 2:
        return "low"
    # "medium": everything else
    return "medium"