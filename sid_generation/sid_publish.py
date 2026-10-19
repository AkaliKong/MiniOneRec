"""Publish a complete catalog directory with one atomic namespace change.

Existing directories are exchanged on macOS/Linux; unsupported filesystems
fail without modifying the published directory. Locks coordinate SID writers.
"""

from contextlib import contextmanager
import ctypes
import errno
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile


GENERATED_SUFFIXES = (
    ".index.json", ".raw.index.json", ".codes.npy", ".item_ids.json",
    ".tokenizer.npz", ".sid_stats.json", ".manifest.json",
)
DATASET_SUFFIXES = (".item.json", ".train.inter", ".valid.inter", ".test.inter")


def generated_names(dataset):
    return [dataset + suffix for suffix in GENERATED_SUFFIXES]


def _previous_outputs(output, dataset):
    """Only trust ownership of reserved names, never arbitrary manifest paths."""
    manifest_path = output / f"{dataset}.manifest.json"
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read previous manifest: {manifest_path}; use a new output directory") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("outputs"), list):
        raise ValueError("previous manifest must contain an outputs list; use a new output directory")
    reserved = {dataset + suffix for suffix in GENERATED_SUFFIXES + DATASET_SUFFIXES}
    return {name for name in manifest["outputs"] if isinstance(name, str) and name in reserved}


def validate_publication(output, dataset, names, overwrite, protected):
    """Preflight inputs and all reserved targets, including obsolete splits."""
    output = Path(output)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("output must be a regular directory, not a file or symbolic link")
    reserved = {dataset + suffix for suffix in GENERATED_SUFFIXES + DATASET_SUFFIXES}
    if len(set(names)) != len(names) or not set(names) <= reserved:
        raise ValueError("unexpected or duplicate catalog output names")
    protected = {Path(path).resolve() for path in protected}
    for name in reserved:
        target = output / name
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"reserved output must be a regular file: {target}")
        if target.resolve() in protected:
            raise ValueError(f"output would replace an input: {target}")
        if name in names and target.exists() and not overwrite:
            raise FileExistsError(f"output already exists: {target}; choose another directory or use --overwrite")
    previous = _previous_outputs(output, dataset)
    obsolete = previous - set(names)
    # A companion file outside a recorded previous export has unknown ownership.
    # Refuse to preserve a stale dataset silently or delete user-owned data.
    for suffix in DATASET_SUFFIXES:
        name = dataset + suffix
        if (output / name).exists() and name not in names and name not in previous:
            raise ValueError(f"untracked dataset companion would remain in export: {output / name}; "
                             "supply --dataset-dir or choose a new output directory")
    return obsolete


@contextmanager
def _writer_lock(output):
    """Lock lives beside the directory, so exchange cannot replace its inode."""
    lock_path = output.parent / f".{output.name}.sid.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        elif os.name == "nt":
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        else:
            raise RuntimeError("catalog publication requires filesystem writer locks")
        yield
    finally:
        # Closing releases the lock even after exceptions/process termination.
        # Keep the inode: unlinking an advisory lock allows competing lock inodes.
        os.close(fd)


def _exchange_directories(staged, output):
    """Swap two existing directories, or leave both unchanged on failure."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        operation = libc.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(os.fsencode(staged), os.fsencode(output), 0x2)  # RENAME_SWAP
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        operation = libc.renameat2
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        result = operation(-100, os.fsencode(staged), -100, os.fsencode(output), 0x2)  # EXCHANGE
    else:
        raise RuntimeError("atomic directory replacement is unavailable on this platform; "
                           "choose a new output directory")
    if result:
        code = ctypes.get_errno()
        if code in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            raise RuntimeError("the filesystem cannot atomically replace this catalog directory; "
                               "choose a new output directory")
        raise OSError(code, os.strerror(code), str(output))


def publish_catalog(payload, output, dataset, names, overwrite, protected):
    """Merge unrelated files, drop obsolete owned files, publish in one step.

    No writes to the public directory occur before commit. This protects against
    process failure, not power loss or noncooperating edits to the output tree.
    Readers needing a multi-open snapshot must pin the directory or verify the
    manifest hashes if a publication can occur while they are reading.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Resolve only the parent: do not silently follow an output symlink.
    output = output.parent.resolve() / output.name
    with _writer_lock(output):
        obsolete = validate_publication(output, dataset, names, overwrite, protected)
        working = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}-publish-"))
        staged = working / "catalog"
        committed = False
        try:
            if output.exists():
                shutil.copytree(output, staged, symlinks=True)
            else:
                staged.mkdir()
            for name in obsolete:
                (staged / name).unlink(missing_ok=True)
            for name in names:
                shutil.copy2(Path(payload) / name, staged / name)
            if output.exists():
                _exchange_directories(staged, output)
            else:
                # The cooperating writer lock makes the initial create safe.
                os.replace(staged, output)
            committed = True
        finally:
            # After exchange, staged contains the *old* directory. Cleanup is
            # outside the commit: never report an already-published export as
            # failed merely because an old temporary directory cannot be removed.
            try:
                shutil.rmtree(working)
            except OSError:
                if not committed:
                    raise
                logging.warning("catalog published; could not remove old snapshot %s", working)
