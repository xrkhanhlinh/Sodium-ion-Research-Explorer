"""Verify original experiment manifests after explicit documentation cleanup.

Only archived code/documents may satisfy historical hashes. Data/model integrity
always requires the live artifact and can never fall back to an archive.
"""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive_eligible(name):
    path = Path(name)
    if len(path.parts) == 1 and path.name.startswith('requirements') and path.suffix == '.txt':
        return True
    return (path.suffix in {'.py', '.md', '.tex', '.pdf', '.ps1', '.log'} and
            (name.startswith(('src/', 'scripts/', 'tests/', 'docs/', 'presentation/')) or
             (name.startswith('outputs/') and path.suffix == '.md') or
             (len(path.parts) == 1 and path.suffix == '.py')))


def archived_match(name, expected, root=ROOT):
    root = Path(root).resolve()
    if not archive_eligible(name):
        return False
    for manifest_path in sorted((root / 'archive').glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entry = manifest.get('files', {}).get(name)
        if entry is None or entry['sha256'] != expected:
            continue
        bundle = (manifest_path.parent / manifest['archive']).resolve()
        bundle.relative_to(root / 'archive')
        if sha256(bundle) != manifest['archive_sha256']:
            raise ValueError('Historical archive changed')
        with zipfile.ZipFile(bundle) as archive:
            return hashlib.sha256(archive.read(name)).hexdigest() == expected
    return False


def verify_snapshot(expected, root=ROOT):
    root = Path(root).resolve()
    live, archived, errors = [], [], []
    for name, checksum in expected.items():
        path = (root / name).resolve()
        path.relative_to(root)
        if path.is_file() and sha256(path) == checksum:
            live.append(name)
        elif archived_match(name, checksum, root):
            archived.append(name)
        else:
            errors.append(name)
    if errors:
        raise ValueError(f'Unverified snapshot artifacts: {errors}')
    return dict(checked=len(expected), unchanged_live=len(live), archived_versions=len(archived),
                archived_paths=archived, status='PASS_LIVE_DATA_AND_ARCHIVED_HISTORY')
