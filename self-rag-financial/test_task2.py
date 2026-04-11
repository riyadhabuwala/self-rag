from app.rag.extractor import FinancialExtractor
e = FinancialExtractor()

sample = """
Apple Inc. (NASDAQ: AAPL) reported total net revenue of \$394.3 billion
for the fiscal year ended September 30, 2023. Net income was \$96.9B,
with earnings per share (EPS) of \$6.13. Gross margin improved to 44.1%
in FY2023 compared to 43.3% in FY2022. The company maintained strong
free cash flow of \$99.6 billion. Under SEC and GAAP reporting standards,
the company disclosed material risk factors in this Form 10-K filing.
"""
print(repr(sample))
result = e.extract_financial_entities(sample)
print("Entities found:")
for k, v in result.items():
    print(f"  {k}: {v}")
assert any('394' in a for a in result['amounts']), 'Revenue amount not found'
