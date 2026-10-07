"""Primitives that make a finished eval durable: atomic writes, a verified copy, one event line per eval."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


class DriveCopyError(RuntimeError):
    """The Drive copy of a result could not be verified. The local file is kept."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_write_text(path, text, encoding='utf-8'):
    """Write via a temp file in the same folder, fsync, then atomic replace: readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding=encoding, newline='') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verified_copy(src, dst, retries=3, sleep=time.sleep):
    """Copy src to dst and prove the copy is identical (SHA-256). A no-op if dst already matches."""
    src, dst = Path(src), Path(dst)
    want = sha256(src)
    if dst.exists() and sha256(dst) == want:
        return dst
    last = None
    for attempt in range(1, retries + 1):
        tmp = dst.with_name(dst.name + '.tmp')
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tmp)
            with open(tmp, 'r+b') as f:
                os.fsync(f.fileno())
            if sha256(tmp) == want:                      # verify BEFORE the copy replaces anything
                os.replace(tmp, dst)
                if sha256(dst) == want:
                    return dst
                last = 'checksum mismatch after replace'
            else:
                last = 'checksum mismatch after copy'
                tmp.unlink(missing_ok=True)
        except OSError as exc:
            last = exc
        if attempt < retries:
            sleep(2 ** attempt)
    raise DriveCopyError(f'could not copy {src} -> {dst} after {retries} attempts ({last}). Local file kept.')


def log_event(cfg, event: dict):
    """Append one JSON line to this shard's log. Best effort: a failure warns and never undoes the eval."""
    line = json.dumps({'time': utc_now(), 'shard': cfg.shard, **event}, ensure_ascii=False, default=str)
    try:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        print(f'  (event log failed: {exc}); event was: {line}')
