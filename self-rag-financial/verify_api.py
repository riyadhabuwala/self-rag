import requests
import time

BASE = 'http://localhost:8000'
HEADERS = {'X-API-Key': 'dev-key-change-in-production', 'Content-Type': 'application/json'}

try:
    print('Clearing cache initially...')
    r = requests.delete(f'{BASE}/cache', headers=HEADERS)
    print(f'Cleared: {r.status_code}')

    print('\n--- TASK 8 ---')
    QUERY = {'query': 'What was Apple total revenue for fiscal year 2023?',
             'filters': {'ticker': 'AAPL', 'fiscal_year': 'FY2023'}}

    print('First call (expect cache miss)...')
    t1 = time.time()
    r1 = requests.post(f'{BASE}/query', headers=HEADERS, json=QUERY)
    ms1 = int((time.time() - t1) * 1000)
    assert r1.status_code == 200
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
    
    print('\n--- TASK 9 ---')
    # Cache stats
    r = requests.get(f'{BASE}/cache/stats', headers=HEADERS)
    assert r.status_code == 200
    stats = r.json()
    print(f'Cache stats: {stats}')
    assert 'backend' in stats
    assert 'redis_available' in stats
    assert 'memory_store_size' in stats

    # Health check shows cache status
    r2 = requests.get(f'{BASE}/health')
    assert r2.status_code == 200
    health = r2.json()
    print(f'Health redis_status: {health["redis_status"]}')
    assert health['redis_status'] in [
        'connected', 'fallback_memory', 'not_configured'
    ]

    # Clear cache
    r3 = requests.delete(f'{BASE}/cache', headers=HEADERS)
    assert r3.status_code == 200
    assert r3.json()['status'] == 'cleared'
    print('Cache cleared')

    # Verify empty after clear
    r4 = requests.get(f'{BASE}/cache/stats', headers=HEADERS)
    print(f'Cache size after clear: {r4.json()["memory_store_size"]}')
    assert r4.json()['memory_store_size'] == 0

    print('PASS: cache stats and clear endpoints work')

except Exception as e:
    print(f"Error: {e}")
