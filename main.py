#!/usr/bin/env python3
"""One Line, One Chill — write a single horror sentence to entry.txt.

The program is deterministic and stdlib-only. It checks the entry
against the competition gate (UTF-8 text, 1-200 characters, not only
whitespace, a single sentence, no control characters, no trailing
newline) and then writes it atomically: a temp file in the same
directory, fsync of file and directory, read-back verification, and
finally ``os.replace`` onto ``entry.txt`` with mode 0644.

How the pieces fit together:

* :func:`validate_entry` is the gate. It is built from small checks
  (``_check_length``, ``_check_printable``, ``_check_shape``,
  ``_check_single_sentence``, ``_encode_utf8``) so each rule can be
  read — and reasoned about — on its own, without any filesystem
  work getting in the way.
* :func:`resolve_output_dir` decides where ``entry.txt`` goes. A
  non-blank ``MAKEFUN_OUTPUT_DIR`` is treated as a request, not a
  hint: the directory is created (parents included) when it does not
  exist yet, and a failure to create it fails the run loudly. Only
  a missing or blank variable falls back to ``/output``. The program
  never silently writes somewhere other than the directory it was
  told to use.
* :func:`cleanup_stale_temps` sweeps temp files left behind by an
  earlier crashed run (``.entry-*``), best-effort, with a short
  retry, so a dirty output directory never blocks a fresh run. Only
  files older than one hour are removed, so a concurrent live run
  is never harmed.
* :func:`write_atomically` does the write itself in two stages:
  ``_create_verified_temp`` builds and verifies the temp file, and
  ``_publish_temp`` renames it onto ``entry.txt`` and forces mode
  0644. On any failure the temp file is removed (never the output
  directory, never a pre-existing ``entry.txt``) and the original
  exception propagates.
* Every failure is an explicit ``raise`` with a plain message. There
  are no ``assert`` statements anywhere: asserts can be stripped
  with ``python -O`` and must never guard a competition gate.
"""

import os
import re
import stat
import sys
import tempfile
import time

#: The entry. 129 characters, one sentence, no trailing newline.
#: Concrete and specific (3 a.m., baby monitor, nursery camera), the
#: dread outlasts the line (she is still rocking the crib), and the
#: last word lands the blow. It avoids the judge's named cliches:
#: no mirror, no doll, no basement.
ENTRY = (
    "At 3 a.m. the baby monitor whispered my name in my dead mother's "
    "voice, and the nursery camera showed her rocking the empty crib."
)

DEFAULT_OUTPUT_DIR = "/output"
TEMP_PREFIX = ".entry-"
TEMP_SUFFIX = ".tmp"
EXPECTED_MODE = 0o644

#: A sentence terminator followed by a space and a capital letter
#: marks a second sentence. Abbreviations such as "a.m." are followed
#: by lowercase text, so they do not trip this check.
_SECOND_SENTENCE_RE = re.compile(r"[.!?]\s+[A-Z]")

#: A temp file older than this is stale and safe to remove.
STALE_TEMP_MAX_AGE_SECONDS = 3600

#: How many times to retry a best-effort stale-temp cleanup.
CLEANUP_ATTEMPTS = 3

#: Pause between cleanup retries, in seconds.
CLEANUP_RETRY_DELAY_SECONDS = 0.1


def resolve_output_dir() -> str:
    """Return the directory ``entry.txt`` will be written into.

    When ``MAKEFUN_OUTPUT_DIR`` names a non-blank path, that path is
    the destination: it is created with ``os.makedirs`` (parents
    included) if it does not exist yet, and verified to be a directory
    afterwards. A path that cannot be created — a file in the way, a
    permission problem — raises :class:`OSError` instead of failing
    over to somewhere else, because writing the entry to a directory
    nobody asked for would silently misdirect the run. Only a
    missing, empty, or whitespace-only variable falls back to
    :data:`DEFAULT_OUTPUT_DIR` (``/output``).
    """
    raw = os.environ.get("MAKEFUN_OUTPUT_DIR")
    candidate = raw.strip() if raw is not None else ""
    if not candidate:
        return DEFAULT_OUTPUT_DIR
    # A named directory is a request, not a hint: create it rather
    # than quietly writing elsewhere.
    os.makedirs(candidate, exist_ok=True)
    if not os.path.isdir(candidate):
        raise OSError(
            f"MAKEFUN_OUTPUT_DIR is not a directory: {candidate!r}"
        )
    return candidate


def _check_length(text: str) -> None:
    """Reject text outside the 1-200 character gate, or all whitespace."""
    if not 1 <= len(text) <= 200:
        raise ValueError(
            f"entry must be 1-200 characters, got {len(text)}"
        )
    if not text.strip():
        raise ValueError("entry must not be only whitespace")


def _check_printable(text: str) -> None:
    """Reject any control character in the entry text."""
    for ch in text:
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            raise ValueError("control character in entry text")


def _check_shape(text: str) -> None:
    """Reject leading/trailing whitespace and a missing terminator.

    The entry must arrive exactly as it will be written: no padding
    and no trailing newline, ending in sentence punctuation.
    """
    if text != text.strip():
        raise ValueError("entry must not have leading/trailing whitespace")
    if text.endswith("\n"):
        raise ValueError("entry must not end with a newline")
    if text[-1] not in ".!?":
        raise ValueError("entry must end with terminal punctuation")


def _check_single_sentence(text: str) -> None:
    """Reject text that holds more than one sentence."""
    if _SECOND_SENTENCE_RE.search(text):
        raise ValueError("entry must be a single sentence")


def _encode_utf8(text: str) -> bytes:
    """Encode *text* as UTF-8, with a round-trip check it survived."""
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"entry is not valid UTF-8 text: {exc}") from exc
    if data.decode("utf-8") != text:
        raise ValueError("entry failed UTF-8 round-trip check")
    return data


def validate_entry(text: str) -> bytes:
    """Validate *text* against the competition gate; return its bytes.

    Runs the checks in order — length, printable characters, shape,
    single sentence, UTF-8 encoding — and raises :class:`ValueError`
    with a specific message on the first failure so the caller can
    report exactly which rule broke.
    """
    if not isinstance(text, str):
        raise ValueError("entry must be a string")
    _check_length(text)
    _check_printable(text)
    _check_shape(text)
    _check_single_sentence(text)
    return _encode_utf8(text)


def _is_stale_temp(path: str, now: float) -> bool:
    """True when *path* is a temp file old enough to remove safely.

    A file younger than :data:`STALE_TEMP_MAX_AGE_SECONDS` may belong
    to a concurrent live run and is never touched; a file that cannot
    be stat'ed is treated as not stale.
    """
    try:
        age = now - os.stat(path).st_mtime
    except OSError:
        return False
    return age > STALE_TEMP_MAX_AGE_SECONDS


def _remove_with_retries(path: str) -> bool:
    """Delete *path*, retrying briefly; True when it is gone."""
    for _ in range(CLEANUP_ATTEMPTS):
        try:
            os.unlink(path)
        except FileNotFoundError:
            return True
        except OSError:
            time.sleep(CLEANUP_RETRY_DELAY_SECONDS)
            continue
        else:
            return True
    return False


def cleanup_stale_temps(out_dir: str) -> int:
    """Remove stale ``.entry-*.tmp`` files from *out_dir*, best-effort.

    A file that still cannot be removed is left in place and the run
    continues — a leftover temp file must never fail an otherwise
    valid entry. Returns the number of files actually removed.
    """
    removed = 0
    now = time.time()
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0
    for name in names:
        if not (
            name.startswith(TEMP_PREFIX) and name.endswith(TEMP_SUFFIX)
        ):
            continue
        path = os.path.join(out_dir, name)
        if not _is_stale_temp(path, now):
            continue
        if _remove_with_retries(path):
            removed += 1
    return removed


def _fsync_dir(path: str) -> None:
    """Flush directory *path* to disk so a rename inside it is durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unlink_quietly(path: str) -> None:
    """Delete *path* if it exists; ignore every error while doing so."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _create_verified_temp(out_dir: str, data: bytes) -> str:
    """Build a temp file holding *data* and verify it by reading back.

    The temp file is created with ``O_EXCL`` semantics (via
    :func:`tempfile.mkstemp`), written, fsynced, forced to mode 0644,
    and read back before the caller renames it. Returns the temp
    path; the caller owns deleting it on failure.
    """
    fd, tmp_path = tempfile.mkstemp(
        dir=out_dir, prefix=TEMP_PREFIX, suffix=TEMP_SUFFIX
    )
    try:
        os.chmod(tmp_path, EXPECTED_MODE)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, EXPECTED_MODE)
        with open(tmp_path, "rb") as fh:
            if fh.read() != data:
                raise ValueError("read-back verification failed")
    except BaseException:
        _unlink_quietly(tmp_path)
        raise
    return tmp_path


def _publish_temp(tmp_path: str, final_path: str, out_dir: str) -> None:
    """Atomically rename *tmp_path* onto *final_path* at mode 0644.

    The directory is fsynced after the rename and again after the
    final chmod, and the resulting mode is verified, so the published
    file is both durable and exactly 0644.
    """
    os.replace(tmp_path, final_path)
    _fsync_dir(out_dir)
    os.chmod(final_path, EXPECTED_MODE)
    _fsync_dir(out_dir)
    mode = stat.S_IMODE(os.stat(final_path).st_mode)
    if mode != EXPECTED_MODE:
        raise ValueError(f"unexpected entry.txt mode: {oct(mode)}")


def write_atomically(out_dir: str, data: bytes) -> str:
    """Write *data* to ``entry.txt`` inside *out_dir*, atomically.

    A temp file is created in the same directory, written, fsynced,
    and read back for verification before ``os.replace`` moves it
    onto ``entry.txt``. On any failure the temp file is removed
    (never the output directory, never a pre-existing ``entry.txt``)
    and the original exception propagates. Returns the final path.
    """
    final_path = os.path.join(out_dir, "entry.txt")
    tmp_path = _create_verified_temp(out_dir, data)
    try:
        _publish_temp(tmp_path, final_path, out_dir)
    except BaseException:
        # Never leave a stale temp file behind; never delete the
        # output directory or a pre-existing entry.txt.
        _unlink_quietly(tmp_path)
        raise
    return final_path


def main() -> int:
    """Validate the entry and write it to ``entry.txt``; return 0."""
    try:
        data = validate_entry(ENTRY)
    except ValueError as exc:
        raise SystemExit(f"invalid entry: {exc}") from exc

    try:
        out_dir = resolve_output_dir()
    except OSError as exc:
        raise SystemExit(f"cannot use output directory: {exc}") from exc
    if not os.path.isdir(out_dir):
        raise SystemExit(f"output directory does not exist: {out_dir}")

    cleanup_stale_temps(out_dir)
    try:
        write_atomically(out_dir, data)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"failed to write entry.txt: {exc}") from exc
    return 0


if __name__ == "__main__":
    sys.exit(main())
