import requests
import concurrent.futures

BASE = 'http://localhost:8000'
HEADERS = {
    'X-API-Key': 'dev-key-change-in-production',
    'Content-Type': 'application/json'
}

queries = [
    'What was Apple revenue in FY2023?',
    'What was Apple net income in FY2023?',
    'What were Apple main risk factors in FY2023?',
]

def run_query(q):
    r = requests.post(f'{BASE}/query', headers=HEADERS,
        json={'query': q,
              'filters': {'ticker': 'AAPL', 'fiscal_year': 'FY2023'}})
    return r.status_code, r.json().get('confidence', 'N/A')

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    futures = [ex.submit(run_query, q) for q in queries]
    results = [f.result() for f in
               concurrent.futures.as_completed(futures)]

for status, confidence in results:
    print(f'  status={status} confidence={confidence}')
    assert status == 200, f'Expected 200, got {status}'

print('PASS: 3 concurrent requests all returned 200')
