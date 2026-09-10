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


def test_python_sanitized_assignment_not_detected():
    code = """
user_input = input()
safe_input = sanitize(user_input)
os.system(safe_input)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert len(vulnerabilities) == 0


def test_python_unsanitized_assignment_detected():
    code = """
user_input = input()
command = user_input
os.system(command)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "command"
    assert vulnerability["sink"] == "os.system"


def test_javascript_sanitized_assignment_not_detected():
    code = """
const userInput = req.query.cmd;
const safeInput = sanitize(userInput);
child_process.exec(safeInput);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert len(vulnerabilities) == 0


def test_javascript_unsanitized_assignment_detected():
    code = """
const userInput = req.query.cmd;
const command = userInput;
child_process.exec(command);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "command"
    assert vulnerability["sink"] == "child_process.exec"


def test_java_sanitized_assignment_not_detected():
    code = """
String userInput = request.getParameter("cmd");
String safeInput = sanitize(userInput);
Runtime.getRuntime().exec(safeInput);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "java",
    )

    assert len(vulnerabilities) == 0


def test_java_unsanitized_assignment_detected():
    code = """
String userInput = request.getParameter("cmd");
String command = userInput;
Runtime.getRuntime().exec(command);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "java",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "command"
    assert vulnerability["sink"] == "Runtime.getRuntime().exec"


def test_python_deep_taint_propagation():
    code = """
user_input = input()
a = user_input
b = a
c = b
os.system(c)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "c"
    assert vulnerability["sink"] == "os.system"


def test_javascript_deep_taint_propagation():
    code = """
const userInput = req.query.cmd;
const a = userInput;
const b = a;
const c = b;
child_process.exec(c);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "c"
    assert vulnerability["sink"] == "child_process.exec"


def test_java_deep_taint_propagation():
    code = """
String userInput = request.getParameter("cmd");
String a = userInput;
String b = a;
String c = b;
Runtime.getRuntime().exec(c);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "java",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "c"
    assert vulnerability["sink"] == "Runtime.getRuntime().exec"


def test_python_expression_taint_propagation():
    code = """
user_input = input()
command = "ping " + user_input
os.system(command)
"""

    vulnerabilities = detect_taint_flows(
        code,
        "python",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "command"
    assert vulnerability["sink"] == "os.system"


def test_javascript_expression_taint_propagation():
    code = """
const userInput = req.query.cmd;
const command = "ping " + userInput;
child_process.exec(command);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "javascript",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "command"
    assert vulnerability["sink"] == "child_process.exec"


def test_java_expression_taint_propagation():
    code = """
String userInput = request.getParameter("cmd");
String command = "ping " + userInput;
Runtime.getRuntime().exec(command);
"""

    vulnerabilities = detect_taint_flows(
        code,
        "java",
    )

    assert len(vulnerabilities) == 1

    vulnerability = vulnerabilities[0]

    assert vulnerability["type"] == "TAINT_FLOW"
    assert vulnerability["source_variable"] == "command"
    assert vulnerability["sink"] == "Runtime.getRuntime().exec"
