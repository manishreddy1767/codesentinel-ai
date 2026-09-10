from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_endpoint():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200


def test_analyze_python_code():

    code = """
user_input = input()
eval(user_input)
"""

    response = client.post(
        "/analyze",
        json={
            "code": code,
            "language": "python",
            "filename": "test.py",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "analysis_id" in data

    assert data["language"] == "python"

    assert data["filename"] == "test.py"

    assert "security_risk" in data

    assert "vulnerability_summary" in data

    assert "vulnerabilities" in data

    assert "cpg_metadata" in data


def test_analyze_sql_injection():

    code = """
user_id = input()
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
"""

    response = client.post(
        "/analyze",
        json={
            "code": code,
            "language": "python",
        },
    )

    assert response.status_code == 200

    data = response.json()

    vulnerability_types = [
        vulnerability["type"]
        for vulnerability in data["vulnerabilities"]
    ]

    assert "SQL_INJECTION" in vulnerability_types


def test_analyze_language_alias():

    response = client.post(
        "/analyze",
        json={
            "code": "print(1)",
            "language": "PY",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["language"] == "python"


def test_analyze_invalid_language():

    response = client.post(
        "/analyze",
        json={
            "code": "puts 'hello'",
            "language": "ruby",
        },
    )

    assert response.status_code == 422


def test_analyze_empty_code():

    response = client.post(
        "/analyze",
        json={
            "code": "",
            "language": "python",
        },
    )

    assert response.status_code == 422


def test_analyze_missing_code():

    response = client.post(
        "/analyze",
        json={
            "language": "python",
        },
    )

    assert response.status_code == 422
