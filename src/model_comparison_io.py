"""Atomic experiment writes tolerant of transient Windows sharing violations."""
import json
import os
from pathlib import Path
import tempfile
import time


def replace_with_retry(source, destination, *, attempts=20, initial_delay=.02):
    """Readers/antivirus may temporarily deny replacement on Windows.

    Never truncate the existing committed file. A permanently denied replace
    raises with the pending temporary file preserved for diagnosis/recovery.
    """
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(.5, initial_delay * 2 ** attempt))


def _temporary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(prefix='.' + path.name + '.', suffix='.tmp',
        dir=path.parent, mode='w', encoding='utf-8', delete=False)
    return stream


def write_json(path, value):
    path = Path(path)
    with _temporary(path) as stream:
        temp = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temp, path)


def write_csv(path, frame):
    path = Path(path)
    with _temporary(path) as stream:
        temp = Path(stream.name)
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    replace_with_retry(temp, path)
