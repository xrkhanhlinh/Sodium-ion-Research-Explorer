"""Current application status; does not rewrite project reports."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'outputs/search_upgrade_20261018'

def current_status():
    path = EXPERIMENT / 'run_status.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'status': 'NOT_RUN', 'quality_status': 'PENDING_INDEPENDENT_LABELS'}

if __name__ == '__main__':
    print(json.dumps(current_status(), ensure_ascii=False, indent=2))
