"""
Tests for the endpoints and behaviours the frontend depends on.

Covers the request paths `frontend/assets/js/api.js` actually makes, the
safety rules on file upload, CORS for the static-server origin, and a
regression guard against re-introducing unsafe DOM rendering.
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from app.config import settings
from app.main import app
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_JS = REPO_ROOT / "frontend" / "assets" / "js"

client = TestClient(app)

VULNERABLE_C = "void process(char* in){ char buf[64]; strcpy(buf, in); }"
CLEAN_C = "int add(int a, int b) { return a + b; }"


# ----------------------------------------------------------------------
# GET /supported-languages
# ----------------------------------------------------------------------


def test_supported_languages_lists_every_validator_language():
    """The selector must offer exactly what the request validator accepts."""

    from app.schemas.request import SUPPORTED_LANGUAGES

    payload = client.get("/supported-languages").json()
    returned = {language["id"] for language in payload["languages"]}

    assert returned == SUPPORTED_LANGUAGES
    assert payload["count"] == len(SUPPORTED_LANGUAGES)


def test_supported_languages_reports_parser_availability():
    """
    A language whose grammar is missing must be reported as unavailable, not
    omitted - hiding it would make an install problem look intentional.
    """

    payload = client.get("/supported-languages").json()

    for language in payload["languages"]:
        assert "parser_available" in language
        assert isinstance(language["parser_available"], bool)
        assert language["extensions"], f"{language['id']} has no extensions"


# ----------------------------------------------------------------------
# POST /analyze
# ----------------------------------------------------------------------


def test_analyze_returns_findings_for_vulnerable_code():
    response = client.post(
        "/analyze", json={"code": VULNERABLE_C, "language": "c"}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["vulnerabilities"], "strcpy should produce at least one finding"
    assert body["language"] == "c"
    assert 0 <= body["security_risk"]["security_score"] <= 100

    finding = body["vulnerabilities"][0]
    for field in ("type", "severity", "confidence"):
        assert field in finding


def test_analyze_returns_empty_list_not_error_for_clean_code():
    """The empty state is a successful response with zero findings."""

    response = client.post("/analyze", json={"code": CLEAN_C, "language": "c"})

    assert response.status_code == 200
    assert response.json()["vulnerabilities"] == []


def test_analyze_rejects_unsupported_language():
    response = client.post("/analyze", json={"code": "x=1", "language": "cobol"})
    assert response.status_code == 422


def test_analyze_rejects_empty_code():
    response = client.post("/analyze", json={"code": "", "language": "python"})
    assert response.status_code == 422


def test_analyze_survives_malformed_source():
    """
    A syntax error is normal input for a security scanner, not a server fault.
    It must not produce a 500.
    """

    response = client.post(
        "/analyze", json={"code": "void f( { { unterminated", "language": "c"}
    )

    assert response.status_code == 200


# ----------------------------------------------------------------------
# POST /analyze/file
# ----------------------------------------------------------------------


def test_analyze_file_accepts_source_and_infers_language():
    response = client.post(
        "/analyze/file", files={"file": ("vuln.c", VULNERABLE_C.encode())}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "c"
    assert body["filename"] == "vuln.c"
    assert body["vulnerabilities"]


def test_analyze_file_strips_directory_from_filename():
    """A path in the upload name must never survive into the response."""

    response = client.post(
        "/analyze/file",
        files={"file": ("../../etc/passwd.c", CLEAN_C.encode())},
    )

    assert response.status_code == 200
    filename = response.json()["filename"]

    assert filename == "passwd.c"
    assert "/" not in filename and "\\" not in filename and ".." not in filename


def test_analyze_file_rejects_unknown_extension():
    response = client.post(
        "/analyze/file", files={"file": ("payload.exe", b"MZ\x90\x00")}
    )
    assert response.status_code == 400


def test_analyze_file_rejects_binary_content():
    response = client.post(
        "/analyze/file", files={"file": ("x.c", bytes([0xFF, 0xFE, 0x00, 0x01]))}
    )
    assert response.status_code == 400
    assert "UTF-8" in response.json()["detail"]


def test_analyze_file_rejects_empty_file():
    response = client.post("/analyze/file", files={"file": ("x.c", b"   ")})
    assert response.status_code == 400


def test_analyze_file_enforces_size_limit():
    oversize = b"a" * (settings.max_upload_bytes + 10)
    response = client.post("/analyze/file", files={"file": ("x.c", oversize)})
    assert response.status_code == 413


def test_analyze_file_honours_explicit_language_override():
    response = client.post(
        "/analyze/file",
        files={"file": ("snippet.txt", b"def f():\n    pass\n")},
        data={"language": "python"},
    )

    assert response.status_code == 200
    assert response.json()["language"] == "python"


# ----------------------------------------------------------------------
# CORS
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin",
    ["http://localhost:5500", "http://127.0.0.1:5500"],
)
def test_cors_allows_the_static_frontend_origin(origin):
    """5500 is the Live Server default the static frontend is served from."""

    response = client.options(
        "/analyze",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_does_not_allow_an_arbitrary_origin():
    response = client.options(
        "/analyze",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers.get("access-control-allow-origin") != "https://evil.example"


# ----------------------------------------------------------------------
# Frontend regression guards
# ----------------------------------------------------------------------


def test_frontend_never_uses_innerhtml_for_dynamic_content():
    """
    Analyzed code is attacker-controlled. Rendering it with innerHTML would
    let a crafted snippet execute script in the operator's browser, so the
    renderer must stay on textContent.
    """

    import re

    def strip_comments(source: str) -> str:
        """
        Remove // and /* */ comments.

        Needed because the renderer's own documentation says it uses
        textContent "never innerHTML", and a naive substring search flags that
        prose as a violation. Checking code rather than comments keeps the test
        strict where it matters instead of weakening the rule.
        """

        source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
        return re.sub(r"//[^\n]*", "", source)

    offenders = []

    for path in FRONTEND_JS.glob("*.js"):
        code = strip_comments(path.read_text(encoding="utf-8"))
        for keyword in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
            if keyword in code:
                offenders.append(f"{path.name}: {keyword}")

    assert not offenders, f"unsafe DOM APIs found: {offenders}"


def test_frontend_pages_contain_no_hardcoded_findings():
    """
    The pages previously shipped invented results (a CWE-120 buffer overflow
    attributed to a nonexistent 'CodeSentinel R-GCN' model). Findings must come
    from the backend only.
    """

    pages = (REPO_ROOT / "frontend" / "pages").glob("*.html")
    banned = ("R-GCN", "CWE-120", "CWE-787", "Buffer Overflow", "Use After Free")

    offenders = []

    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        for token in banned:
            if token in text:
                offenders.append(f"{page.name}: {token}")

    assert not offenders, f"hardcoded findings still present: {offenders}"


def test_result_pages_load_their_controllers():
    """Each rewritten page must actually wire up its module."""

    expected = {
        "analyze-code.html": "analyze-page.js",
        "analysis-results.html": "results-page.js",
        "vulnerability-details.html": "details-page.js",
        "history.html": "notimpl-history.js",
        "reports.html": "notimpl-reports.js",
        "repository-analysis.html": "notimpl-repository.js",
    }

    for page, script in expected.items():
        text = (REPO_ROOT / "frontend" / "pages" / page).read_text(
            encoding="utf-8", errors="replace"
        )
        assert script in text, f"{page} does not load {script}"
        assert 'id="cs-results"' in text, f"{page} has no results container"
