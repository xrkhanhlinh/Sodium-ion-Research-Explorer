"""Archive verified historical reports/code, then remove obsolete active files.

Only enumerated workspace files may be removed. Every removed byte is first
preserved in a ZIP and SHA-256 checked. Data, models and measured CSV/JSON
artifacts are outside the removal set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'archive/cleanup_20261018'
REMOVE = [
    'src/stage4_common.py', 'src/stage4_index.py', 'src/stage4_encode.py',
    'src/stage4_evaluate.py', 'src/stage4_figures.py', 'src/stage4_complete.py',
    'src/stage5_service.py', 'src/stage5_report.py',
    'scripts/stage4_smoke.py', 'scripts/stage4_continue.py',
    'scripts/stage5_browser.py', 'scripts/stage5_restart_control.py',
    'scripts/review_research_quality.py', 'scripts/complete_research_audit.py',
    'scripts/summarize_search_quality_pilot.py', 'scripts/acceptance_audit.py',
    'tests/test_stage4_unit.py', 'tests/test_stage4_integration.py',
    'tests/test_stage5_app.py', 'tests/test_stage5_backend.py',
]
REPLACE = [
    'app.py', 'src/search_v2_service.py', 'src/stage4_search.py',
    'src/stage4_benchmark.py', 'src/stage4_live_latency.py', 'src/stage4_queries.py',
    'src/stage5_common.py', 'run_stage4.py', 'search_stage4.py',
    'docs/FINAL_RESEARCH_REPORT.md', 'docs/RUN_DEMO.md', 'docs/PROJECT_SPEC.md',
    'docs/PROGRESS.md', 'docs/EXPERIMENT_LOG.md',
    'scripts/benchmark_search_v2.py', 'src/pipeline.py', 'src/stage2.py',
    'src/stage3_common.py', 'src/complete.py', 'src/stage2_complete.py',
    'src/stage3_complete.py', 'run_stage2.py',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    if not path.is_file():
        raise ValueError(f'Expected a workspace file: {relative}')
    return path


def prepare():
    if DEST.exists():
        raise ValueError('Cleanup archive already exists; never overwrite it')
    actions = {p: 'replace' for p in REPLACE}
    actions.update({p: 'remove' for p in REMOVE})
    for p in (ROOT / 'docs').glob('STAGE*_REPORT.md'):
        actions[p.relative_to(ROOT).as_posix()] = 'remove'
    for p in (ROOT / 'outputs/reports').rglob('*.md'):
        actions[p.relative_to(ROOT).as_posix()] = 'remove'
    # The 45-slide bundle describes the pre-v2 search. Preserve it as history,
    # not as a current presentation. A new small README will identify the archive.
    for p in (ROOT / 'presentation').rglob('*'):
        if p.is_file():
            actions[p.relative_to(ROOT).as_posix()] = 'remove'
    for name in actions:
        safe_path(name)
        if name.startswith(('models/', 'data/')):
            raise ValueError('Scientific data/model deletion is prohibited')
    DEST.mkdir(parents=True)
    entries = {name: dict(action=action, sha256=sha(ROOT / name), bytes=(ROOT / name).stat().st_size)
               for name, action in sorted(actions.items())}
    archive = DEST / 'historical_files.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for name in entries:
            bundle.write(ROOT / name, arcname=name)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Corrupt cleanup archive')
        for name, info in entries.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != info['sha256']:
                raise ValueError(f'Archive byte mismatch: {name}')
    manifest = dict(archive='historical_files.zip', archive_sha256=sha(archive),
                    files=entries, removal_applied=False,
                    policy='No data/models/CSV metrics removed. Archive preserves unfavorable and favorable experiments alike.')
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verified archive:', len(entries), 'files;', sum(x['action'] == 'remove' for x in entries.values()), 'scheduled removals')


def apply():
    path = DEST / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    archive = DEST / manifest['archive']
    if sha(archive) != manifest['archive_sha256']:
        raise ValueError('Archive changed')
    targets = [(name, safe_path(name)) for name, info in manifest['files'].items()
               if info['action'] == 'remove']
    with zipfile.ZipFile(archive) as bundle:
        for name, target in targets:
            expected = manifest['files'][name]['sha256']
            if sha(target) != expected or hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError(f'Active file or archive changed since review: {name}')
    # All absolute targets checked inside ROOT, and all backups verified, before
    # any deletion. No recursive directory deletion or cross-shell operation.
    for name, target in targets:
        target.unlink()
    for folder in sorted((ROOT / 'presentation').rglob('*'), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    manifest['removal_applied'] = True
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Removed from active tree:', len(targets), 'verified archived files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'apply'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else apply()
