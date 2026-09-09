from app.services.taint_service import detect_taint_flows


def test_python_eval_taint_flow():
    code = """
user_input = input()
eval(user_input)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert any(
        vulnerability["sink"] == "eval"
        and vulnerability["source_variable"] == "user_input"
        and vulnerability["severity"] == "CRITICAL"
        and vulnerability["cwe"] == "CWE-95"
        for vulnerability in vulnerabilities
    )


def test_python_command_injection_taint_flow():
    code = """
import os

command = input()
os.system(command)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert any(
        vulnerability["sink"] == "os.system"
        and vulnerability["source_variable"] == "command"
        and vulnerability["cwe"] == "CWE-78"
        for vulnerability in vulnerabilities
    )


def test_javascript_eval_taint_flow():
    code = """
const userInput = prompt();
eval(userInput);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert any(
        vulnerability["sink"] == "eval"
        and vulnerability["source_variable"] == "userInput"
        and vulnerability["cwe"] == "CWE-95"
        for vulnerability in vulnerabilities
    )


def test_javascript_command_injection_taint_flow():
    code = """
const command = req.query.cmd;
child_process.exec(command);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert any(
        vulnerability["sink"] == "child_process.exec"
        and vulnerability["source_variable"] == "command"
        and vulnerability["cwe"] == "CWE-78"
        for vulnerability in vulnerabilities
    )


def test_java_command_injection_taint_flow():
    code = """
String command = request.getParameter("cmd");
Runtime.getRuntime().exec(command);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "java",
    )

    assert any(
        vulnerability["sink"] == "Runtime.getRuntime().exec"
        and vulnerability["source_variable"] == "command"
        and vulnerability["cwe"] == "CWE-78"
        for vulnerability in vulnerabilities
    )


def test_python_sql_injection_taint_flow():
    code = """
user_id = input()
query = "SELECT * FROM users WHERE id = " + user_id
cursor.execute(query)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert any(
        vulnerability["sink"] == "execute"
        and vulnerability["source_variable"] == "query"
        and vulnerability["cwe"] == "CWE-89"
        for vulnerability in vulnerabilities
    )


def test_javascript_sql_injection_taint_flow():
    code = """
const userInput = req.query.id;
const query = "SELECT * FROM users WHERE id = " + userInput;
db.query(query);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert any(
        vulnerability["sink"] == "query"
        and vulnerability["source_variable"] == "query"
        and vulnerability["cwe"] == "CWE-89"
        for vulnerability in vulnerabilities
    )


def test_java_sql_injection_taint_flow():
    code = """
String userId = request.getParameter("id");
String query = "SELECT * FROM users WHERE id = " + userId;
statement.executeQuery(query);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "java",
    )

    assert any(
        vulnerability["sink"] == "executeQuery"
        and vulnerability["source_variable"] == "query"
        and vulnerability["cwe"] == "CWE-89"
        for vulnerability in vulnerabilities
    )
