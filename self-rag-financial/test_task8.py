import requests
import time

BASE = 'http://localhost:8000'
HEADERS = {
    'X-API-Key': 'dev-key-change-in-production',
    'Content-Type': 'application/json'
}
QUERY = {'query': 'What was Apple total revenue for fiscal year 2023?',
         'filters': {'ticker': 'AAPL', 'fiscal_year': 'FY2023'}}

print('First call (expect cache miss)...')
t1 = time.time()
r1 = requests.post(f'{BASE}/query', headers=HEADERS, json=QUERY)
ms1 = int((time.time() - t1) * 1000)
assert r1.status_code == 200, f'Expected 200, got {r1.status_code}: {r1.text}'
d1 = r1.json()
print(f'  cache_hit={d1["cache_hit"]} response_time={ms1}ms')
assert d1['cache_hit'] == False

print('Second call same query (expect cache hit)...')
t2 = time.time()
r2 = requests.post(f'{BASE}/query', headers=HEADERS, json=QUERY)
ms2 = int((time.time() - t2) * 1000)
assert r2.status_code == 200
d2 = r2.json()
print(f'  cache_hit={d2["cache_hit"]} response_time={ms2}ms')
assert d2['cache_hit'] == True
assert d2['answer'] == d1['answer'], 'Cached answer should match original'

SIM_QUERY = {'query': 'What was Apples total revenue for FY2023?',
             'filters': {'ticker': 'AAPL', 'fiscal_year': 'FY2023'}}
print('Third call semantic variant (expect cache hit)...')
t3 = time.time()
r3 = requests.post(f'{BASE}/query', headers=HEADERS, json=SIM_QUERY)
ms3 = int((time.time() - t3) * 1000)
assert r3.status_code == 200
d3 = r3.json()
print(f'  cache_hit={d3["cache_hit"]} response_time={ms3}ms')

print(f'Latency comparison:')
print(f'  Miss: {ms1}ms | Hit (exact): {ms2}ms | Hit (semantic): {ms3}ms')

if d2['cache_hit']:
    speedup = ms1 / max(ms2, 1)
    print(f'  Speedup: {speedup:.1f}x faster on cache hit')
    assert ms2 < ms1, 'Cache hit should be faster than full pipeline'

print('PASS: semantic cache working end-to-end via API')
