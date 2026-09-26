"""
Pytest path setup for the whole repository.

The two test suites import from different roots:

    backend/tests/test_api.py etc.  ->  `import app.services...`   needs backend/
    backend/tests/test_ml_*.py      ->  `import backend.ml...`     needs the repo root

So neither working directory satisfies both, and running the full suite from a
single place used to fail collection with ModuleNotFoundError no matter which
directory you picked. Putting both roots on sys.path here makes
`python -m pytest backend/tests/` work from the repository root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

for path in (ROOT, BACKEND):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)


# `legacy/` holds a superseded parallel implementation kept for reference only
# (see legacy/README.md). Its tests target an API that no longer exists, so
# collecting them would report failures for code that is intentionally retired.
collect_ignore_glob = ["legacy/*"]
