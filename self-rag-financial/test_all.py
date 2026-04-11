from app.graph.builder import build_graph, run_query
import sys

print("Building graph...")
graph = build_graph()

# Task 3
print("\n--- TASK 3 ---")
result = run_query(query="Hello, what is your name?", graph=graph)
print("Direct answer result:")
print(f"  needs_retrieval: {result['needs_retrieval']}")
print(f"  answer: {result['answer'][:100]}")
print(f"  confidence: {result['confidence']}")
print(f"  response_time_ms: {result['response_time_ms']}")
assert result['needs_retrieval'] == False, "Should not need retrieval"
assert result['answer'] != "", "Answer should not be empty"
print("PASS: direct answer path works")

# Task 4
print("\n--- TASK 4 ---")
result = run_query(query="What was Apple total revenue for fiscal year 2023?", graph=graph)
print("Full retrieval result:")
print(f"  needs_retrieval: {result['needs_retrieval']}")
print(f"  retrieved_chunks: {len(result['retrieved_chunks'])}")
print(f"  relevant_chunks: {len(result['relevant_chunks'])}")
print(f"  retry_count: {result['retry_count']}")
print(f"  groundedness: {result['groundedness']}")
print(f"  usefulness_score: {result['usefulness_score']}")
print(f"  confidence: {result['confidence']}")
print(f"  unsupported_claims: {result['unsupported_claims']}")
print(f"  sources: {len(result['sources'])} sources")
print(f"  response_time_ms: {result['response_time_ms']}ms\n")
print("Answer:")
print(result['answer'])
assert result['needs_retrieval'] == True
assert len(result['retrieved_chunks']) > 0
assert len(result['relevant_chunks']) > 0
assert result['answer'] != ''
assert result['groundedness'] in ['fully', 'partially', 'no']
assert result['confidence'] in ['high', 'medium', 'low']
assert result['usefulness_score'] >= 1
print("PASS: full retrieval path works end-to-end")

# Task 5
print("\n--- TASK 5 ---")
result = run_query(query="What were the main risk factors disclosed?", filters={"ticker": "AAPL", "fiscal_year": "FY2023"}, graph=graph)
print("Filtered retrieval result:")
print(f"  relevant_chunks: {len(result['relevant_chunks'])}")
print(f"  confidence: {result['confidence']}")
for chunk in result['relevant_chunks']:
    meta = chunk['metadata']
    assert meta['ticker'] == 'AAPL', f"Wrong ticker: {meta['ticker']}"
print("PASS: all chunks from correct filtered document")

# Task 6
print("\n--- TASK 6 ---")
result = run_query(query="What was Microsoft Azure revenue growth in Q2 2024?", filters={"ticker": "MSFT"}, graph=graph)
print("Graceful failure result:")
print(f"  retry_count: {result['retry_count']}")
print(f"  confidence: {result['confidence']}")
print(f"  answer: {result['answer'][:200]}")
assert result['answer'] != "", "Should return a failure message, not empty"
assert result['confidence'] in ['low', 'medium', 'high']
print("PASS: graceful failure on non-existent document")

# Task 7
print("\n--- TASK 7 ---")
result = run_query(query="xyz performance metrics numbers data", max_retries=2, graph=graph)
print("Retry logic result:")
print(f"  retry_count: {result['retry_count']}")
print(f"  active_query: {result['active_query']}")
print(f"  confidence: {result['confidence']}")
print(f"  answer preview: {result['answer'][:150]}")
print("PASS: retry logic executed without crash")

# Task 8
print("\n--- TASK 8 ---")
result = run_query(query="What was Apple net income in FY2023?", graph=graph)
required_keys = [
    'query', 'session_id', 'filters', 'needs_retrieval', 'query_type',
    'active_query', 'retrieved_chunks', 'relevant_chunks', 'retry_count',
    'answer', 'groundedness', 'groundedness_confidence', 'unsupported_claims',
    'usefulness_score', 'confidence', 'sources', 'response_time_ms',
    'cache_hit', 'failure_reason'
]
print("Checking all state keys present:")
for key in required_keys:
    assert key in result, f"Missing key: {key}"
    print(f"  ? {key}: {str(result[key])[:60]}")
print("\nPASS: all required state keys present in final output")
