"""Contract smoke test against a running FlyRely API.

    FLYRELY_API_URL=http://localhost:8000 python scripts/smoke_test.py

Exits non-zero if /health or /predict does not honour the canonical contract.
"""
import json
import os
import sys
import urllib.request

API = os.environ.get('FLYRELY_API_URL', 'http://localhost:8000').rstrip('/')

PREDICT_FIELDS = {
    'flight_id', 'risk_level', 'delay_probability', 'confidence', 'delay_severity',
    'origin', 'destination', 'departure_time', 'airline', 'weather_available',
    'origin_weather', 'destination_weather', 'risk_factors', 'recommendations',
    'warnings', 'model_version',
}


def call(path, payload=None):
    req = urllib.request.Request(
        f'{API}{path}',
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())


def main():
    print('API:', API)
    status, health = call('/health')
    print('/health', status, health)
    if health.get('status') != 'healthy':
        sys.exit('health is not healthy')

    payload = {'flight_id': 'smoke-1', 'origin': 'JFK', 'destination': 'LAX',
               'departure_time': '2026-11-15T14:30:00', 'airline': 'AA'}
    status, body = call('/predict', payload)
    print('/predict', status, json.dumps(body, indent=2))
    missing = PREDICT_FIELDS - body.keys()
    if missing:
        sys.exit(f'missing fields: {sorted(missing)}')
    if body['flight_id'] != 'smoke-1':
        sys.exit('flight_id not echoed')
    print('OK')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.exit(f'smoke test failed: {e}')
