from app.graph.builder import build_graph, run_query
graph = build_graph()

result = run_query(
    query="Hello, what is your name?",
    graph=graph
)
print("Direct answer result:")
print(f"  needs_retrieval: {result['needs_retrieval']}")
print(f"  answer: {result['answer'][:100]}")
print(f"  confidence: {result['confidence']}")
print(f"  response_time_ms: {result['response_time_ms']}")
assert result['needs_retrieval'] == False, "Should not need retrieval"
assert result['answer'] != "", "Answer should not be empty"
print("PASS: direct answer path works")
