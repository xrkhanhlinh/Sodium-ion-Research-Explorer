import hashlib
import json
import zipfile

import pytest
from src.artifact_history import verify_snapshot


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bundle(root, name='docs/obsolete.md'):
    folder = root / 'archive/cleanup'
    folder.mkdir(parents=True)
    data = b'original evidence'
    path = folder / 'history.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(name, data)
    (folder / 'manifest.json').write_text(json.dumps(dict(
        archive=path.name, archive_sha256=digest(path.read_bytes()),
        files={name: dict(sha256=digest(data))})), encoding='utf-8')
    return data, path


def test_archived_original_report_remains_verifiable(tmp_path):
    data, _ = bundle(tmp_path)
    result = verify_snapshot({'docs/obsolete.md': digest(data)}, tmp_path)
    assert result['unchanged_live'] == 0 and result['archived_versions'] == 1


def test_current_report_can_change_without_rewriting_historical_evidence(tmp_path):
    data, _ = bundle(tmp_path)
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/obsolete.md').write_bytes(b'updated current report')
    assert verify_snapshot({'docs/obsolete.md': digest(data)}, tmp_path)['archived_versions'] == 1


@pytest.mark.parametrize('name', ['data/corpus.parquet', 'models/model.safetensors', 'outputs/tables/metrics.csv'])
def test_missing_scientific_artifact_cannot_be_hidden_in_archive(tmp_path, name):
    data, _ = bundle(tmp_path, name)
    with pytest.raises(ValueError, match='Unverified'):
        verify_snapshot({name: digest(data)}, tmp_path)


def test_tampered_archive_is_rejected(tmp_path):
    data, path = bundle(tmp_path)
    path.write_bytes(b'corrupted archive')
    with pytest.raises(ValueError, match='archive changed'):
        verify_snapshot({'docs/obsolete.md': digest(data)}, tmp_path)


def test_unarchived_missing_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unverified'):
        verify_snapshot({'docs/missing.md': '0' * 64}, tmp_path)


def test_paths_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError):
        verify_snapshot({'../secret.md': '0' * 64}, tmp_path)
