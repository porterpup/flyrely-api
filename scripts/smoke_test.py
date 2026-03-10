import requests, os, sys
API = os.environ.get('FLYRELY_API_URL','https://web-production-ea1e9.up.railway.app')

print('API:', API)
try:
    r = requests.get(f'{API}/health', timeout=10)
    print('/health', r.status_code, r.json())

    payload = {"origin":"JFK","destination":"LAX","departure_time":"2026-03-15T14:30:00","airline":"AA"}
    p = requests.post(f'{API}/predict', json=payload, timeout=15)
    print('/predict', p.status_code)
    try:
        print(p.json())
    except Exception as e:
        print('predict non-json or error', e)
except Exception as e:
    print('smoke test failed', e)
    sys.exit(2)
