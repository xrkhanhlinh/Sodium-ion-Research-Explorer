"""Metadata-only Gemini preflight; credentials never appear in URLs or output."""
import json
import hashlib
from pathlib import Path
import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/retrieval_ai_evaluation_v1'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = ROOT / 'docs/api.env'
    key = dotenv_values(env).get('GEMINI_API_KEY')
    if not key:
        raise SystemExit('Missing GEMINI_API_KEY; value not displayed')
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get('https://generativelanguage.googleapis.com/v1beta/models',
                               headers={'x-goog-api-key': key}, params={'pageSize': 1000}, timeout=45)
    except requests.RequestException as exc:
        print(json.dumps({'status': 'NETWORK_ERROR', 'error_type': type(exc).__name__}))
        return
    body = response.json()
    auth_path=OUT/'authorization.json'
    auth=json.loads(auth_path.read_text(encoding='utf-8')) if auth_path.exists() else {}
    bound=auth.get('credential_sha256')==hashlib.sha256(key.encode()).hexdigest()
    record = {'http_status': response.status_code, 'billing_check': 'USER_ATTESTATION_NOT_API_VERIFIED' if bound else 'PENDING_CURRENT_KEY_ATTESTATION',
              'billing_enabled_user_attestation': False if bound and auth.get('billing_disabled') is True else None,
              'upload_authorized_by_user': bool(bound and auth.get('upload_authorized') is True),
              'pricing_source': 'https://ai.google.dev/gemini-api/docs/pricing',
              'models': [{'name': m['name'], 'methods': m.get('supportedGenerationMethods', []),
                          'version': m.get('version')} for m in body.get('models', [])],
              'error_status': body.get('error', {}).get('status'),
              'error_message': body.get('error', {}).get('message', '').replace(key, '[REDACTED]'),
              'next_page_present': bool(body.get('nextPageToken'))}
    text = json.dumps(record, ensure_ascii=False, indent=2).replace(key, '[REDACTED]')
    (OUT / 'api_preflight.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
