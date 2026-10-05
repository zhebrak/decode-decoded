"""Make servebench and anatomy importable when pytest runs from anywhere without an install, and mark what only the
Linux GPU box can run: planning, tidy and the page run anywhere; collection needs Linux's CPU affinity and the
session script GNU coreutils' timeout."""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

linux_affinity = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"),
                                    reason="pins threads to cores with Linux's CPU affinity: collection, on the box")
gnu_timeout = pytest.mark.skipif(shutil.which("timeout") is None,
                                 reason="scripts/session.sh bounds its uploads with GNU coreutils' timeout")
