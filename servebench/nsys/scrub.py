"""Nsight exports carry the environment of the processes they traced: StringIds rows such as SSH_CONNECTION=... or a
cloud API key, single or as whole NUL-joined environment blocks (a pod's exports held a provider's API key). Never data,
and never to be published. scrub() blanks every such row, in a host-owned copy renamed over the export (a container's
export is root-owned, and could not be written in place); kernel names, NVTX ranges and everything
servebench.nsys.parse reads stay. leaks() is the check that nothing secret-looking is left in a tree about to be
uploaded (scripts/sync_raw.sh runs both, and refuses the upload when anything remains).

    python -m servebench.nsys.scrub <dir or file> ...    exit 1 when anything secret-looking remains
"""
import gzip
import re
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

# NAME=value at the start or after a NUL: an environment entry, or a block of them whose first entry has no value
# (PTXAS_FLAGS\0INCLUDES=...\0SSH_CLIENT=... on the pod); in 349 real exports it matched no kernel name
ENV = re.compile(r"(?:^|\x00)[A-Za-z_][A-Za-z0-9_]*=")
SECRET = re.compile(rb"(_API_KEY|_TOKEN|_SECRET|PASSWORD|SSH_CONNECTION|SSH_CLIENT|PUBLIC_KEY)=")
# the same names as a JSON or YAML key with a value ("HF_TOKEN": "...", HF_TOKEN: ...), upper-case as an environment's
# are (vLLM's own hf_token argument is lower-case and empty), and strings shaped like the tokens themselves: a bare
# prefix (hf_, sk-) matched 15 real files' vLLM arguments (hf_overrides) and task- or disk- names; these shapes, none
KEYED = re.compile(rb'"?\b[A-Z][A-Z0-9_]*(?:_API_KEY|_TOKEN|_SECRET|PASSWORD)"?[ \t]*:[ \t]*"?'
                   rb'(?!null\b|None\b)[^\s",}{]')
TOKENS = re.compile(rb"\bhf_[A-Za-z0-9]{30,}|\bghp_[A-Za-z0-9]{36}|\bgithub_pat_\w{40,}|\bsk-[A-Za-z0-9_-]{20,}"
                    rb"|\bAKIA[0-9A-Z]{16}\b|\bxox[bp]-[A-Za-z0-9-]{20,}")
SKIP = (".nsys-rep",)  # compressed: cannot be scrubbed, so never uploaded (sync_raw.sh excludes them)


def scrub(path):
    """Blank environment entries in a host-owned copy, then atomically replace the export.

    Docker exports may be root-owned and read-only to the uploader. Never modify the original until the copy is
    scrubbed and vacuumed; a failed scrub must leave the original available for retry.
    """
    path = Path(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if not db.execute("select 1 from sqlite_master where type='table' and name='StringIds'").fetchone():
            return 0
        rows = [i for i, v in db.execute("select id, value from StringIds") if v and ENV.search(v)]
        if not rows:
            return 0
        with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as tmp:
            replacement = Path(tmp) / path.name
            clean = sqlite3.connect(replacement)
            try:
                db.backup(clean)
                clean.executemany("update StringIds set value = '' where id = ?", [(i,) for i in rows])
                clean.commit()
                clean.execute("vacuum")  # no secret text left in free pages
                if clean.execute("pragma integrity_check").fetchone() != ("ok",):
                    raise RuntimeError(f"scrub: corrupt SQLite copy of {path}")
            finally:
                clean.close()
            replacement.replace(path)
    return len(rows)


def secret_looking(data):
    return any(rx.search(data) for rx in (SECRET, KEYED, TOKENS))


def contents(path):
    """A file's bytes, a .gz one's decompressed (the chunk arrival files, any compressed log); its raw bytes when it
    does not decompress."""
    data = path.read_bytes()
    if path.name.endswith(".gz"):
        try:
            return gzip.decompress(data)
        except (OSError, EOFError):
            return data
    return data


def leaks(root):
    """Files under root (export or not, .nsys-rep aside) that still hold something secret-looking: an environment
    entry NAME=value, the same name as a JSON or YAML key with a value, or a string shaped like a token; .gz files
    read decompressed. A check on what an upload holds, not a proof that it holds nothing: sync_raw.sh uploads only
    known artifact types, and canary tests keep the patterns honest (tests/test_nsys_parse.py)."""
    root = Path(root)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    return [p for p in files if not p.name.endswith(SKIP) and secret_looking(contents(p))]


def main(argv=None):
    paths = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    blanked = 0
    for p in paths:
        for f in [p] if p.is_file() else sorted(p.rglob("*.sqlite")):
            blanked += scrub(f)
    left = [f for p in paths for f in leaks(p)]
    print(f"scrub: {blanked} environment rows blanked; {len(left)} files still secret-looking")
    for f in left:
        print(f"  {f}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
