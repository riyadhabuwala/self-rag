from app.rag.extractor import FinancialExtractor
from app.graph.builder import build_graph, run_query

e = FinancialExtractor()

print("--- TASK 3: Intent Classification (Advice) ---")
res3 = e.classify_query_intent("Should I buy AAPL stock now?")
print(res3)
assert res3["intent"] == "advice"
assert res3["requires_disclaimer"] is True
print("PASS: Task 3")

print("\n--- TASK 4: Intent Classification (Comparison) ---")
res4 = e.classify_query_intent("How does Apple's revenue compared to Microsoft's?")
print(res4)
assert res4["intent"] == "comparison"
assert res4["requires_disclaimer"] is False
print("PASS: Task 4")

print("\n--- TASK 5: Guardrails Node in Graph ---")
graph = build_graph()
res5 = run_query(query="Should I buy AAPL stock?", graph=graph)
print("Disclaimer:", res5.get("disclaimer"))
assert res5.get("disclaimer") is not None
print("PASS: Task 5")
