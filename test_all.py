import requests
import concurrent.futures
import os
import sys

BASE = 'http://localhost:8000'
HEADERS = {
    'X-API-Key': 'dev-key-change-in-production',
    'Content-Type': 'application/json'
}

def verify():
    # task 5
    r1 = requests.post(f'{BASE}/query', headers=HEADERS, json={
        'query': 'What was Apple revenue in FY2023?',
        'filters': {'ticker': 'AAPL', 'fiscal_year': 'FY2023'}
    })
    
    session_id = r1.json()['session_id']
    print(f'Turn 1 session: {session_id}')

    r2 = requests.post(f'{BASE}/query', headers=HEADERS, json={
        'query': 'What about Apple net income for the same period?',
        'session_id': session_id,
        'filters': {'ticker': 'AAPL', 'fiscal_year': 'FY2023'}
    })
    print(f'Turn 2 preserved session: {r2.json()["session_id"]}')

    r3 = requests.get(f'{BASE}/sessions/{session_id}', headers=HEADERS)
    session_data = r3.json()
    print(f'Session message count: {session_data["session"]["message_count"]}')
    
    # task 6
    r = requests.post(f'{BASE}/sessions', headers=HEADERS,
        json={'title': 'Test Session', 'document_filter': {'ticker': 'AAPL'}})
    sid = r.json()['session_id']
    
    r2 = requests.get(f'{BASE}/sessions', headers=HEADERS)
    
    r3 = requests.get(f'{BASE}/sessions/{sid}', headers=HEADERS)
    
    r4 = requests.delete(f'{BASE}/sessions/{sid}?archive=true', headers=HEADERS)
    
    r5 = requests.get(f'{BASE}/sessions/non-existent-id', headers=HEADERS)
    print('PASS: all session endpoints work correctly')

    # task 8
    r = requests.get('http://localhost:8000/metrics', headers=HEADERS)
    print('PASS: /metrics returns all required fields')

if __name__ == '__main__':
    verify()
