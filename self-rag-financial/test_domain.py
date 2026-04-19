from app.graph.builder import build_graph, run_query

graph = build_graph()

queries = [
    "Should I invest my life savings in AAPL?",
    "What was Apple's total net sales in FY2023?",
    "How does Apple's net income compare to Microsoft's in FY2023?"
]

for q in queries:
    print(f"\n======================================")
    print(f"QUERY: {q}")
    try:
        result = run_query(query=q, graph=graph)
        print(f"\nFILTERS EXTRACTED: {result.get('filters')}")
        print(f"DISCLAIMER ATTACHED: {'Yes' if result.get('disclaimer') else 'No'}")
        print(f"SOURCES USED: {[s.get('document') for s in result.get('sources', [])]}")
        print(f"\nANSWER:\n{result.get('answer')}")
    except Exception as e:
         print(f"Error running query: {e}")
