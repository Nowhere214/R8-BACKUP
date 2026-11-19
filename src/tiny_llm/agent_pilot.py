"""Small, deterministic code-repair agent and isolated test tool."""

from __future__ import annotations

import ast
import json
import re
import signal
import subprocess
import sys
from pathlib import Path
from time import perf_counter


SYSTEM_PROMPT = (
    "You solve small Python programming tasks. Return the complete Python solution "
    "in a single ```python code block. Use the exact function name from the example. "
    "Do not print, read files, use the network, or run commands. Use only the Python "
    "standard library. After test feedback, return a complete corrected solution."
)
ALLOWED_IMPORTS = {
    "math",
    "itertools",
    "collections",
    "functools",
    "heapq",
    "bisect",
    "re",
    "string",
    "statistics",
    "typing",
    "operator",
    "decimal",
    "fractions",
}


def extract_code(response: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return response.strip()


def check_code_policy(code: str) -> None:
    tree = ast.parse(code)
    forbidden = {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name.split(".")[0] not in ALLOWED_IMPORTS for alias in node.names
            ):
                raise ValueError("Only approved standard-library imports are allowed")
        if isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError("Only approved standard-library imports are allowed")
        if isinstance(node, ast.Name) and (
            node.id in forbidden or node.id.startswith("__")
        ):
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attribute access is not allowed")


def sandbox_profile() -> str:
    runtime = json.dumps(str(Path(sys.base_prefix).resolve()))
    return (
        "(version 1) (allow default) (deny network*) (deny file-write*) "
        "(deny process-fork) "
        '(deny file-read-data (require-all (subpath "/Users") '
        f"(require-not (subpath {runtime})))) "
        '(deny file-read-data (subpath "/Volumes") (subpath "/private/tmp") '
        '(subpath "/private/var/folders"))'
    )


RUNNER = r"""
import ast, contextlib, io, json, resource, sys, traceback
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
payload = json.load(sys.stdin)
scope = {"__name__": "solution"}
result = {"status": "passed", "passed": True}
try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for statement in payload["imports"]:
            exec(statement, scope)
        exec(payload["code"], scope)
        for test in payload["tests"]:
            tree = ast.parse(test)
            check = tree.body[0] if len(tree.body) == 1 else None
            comparison = check.test if isinstance(check, ast.Assert) else None
            if (isinstance(comparison, ast.Compare) and len(comparison.ops) == 1
                    and isinstance(comparison.ops[0], ast.Eq)):
                actual = eval(compile(ast.Expression(comparison.left), "<visible-test>", "eval"), scope)
                expected = eval(compile(ast.Expression(comparison.comparators[0]), "<visible-test>", "eval"), scope)
                if actual != expected:
                    raise AssertionError(f"{ast.unparse(comparison.left)} returned {repr(actual)[:200]}; expected {repr(expected)[:200]}")
            else:
                exec(test, scope)
except BaseException as error:
    result = {"status": "failed", "passed": False,
              "feedback": (type(error).__name__ + ": " + str(error))[:500],
              "traceback": traceback.format_exc(limit=3)[-1200:]}
print(json.dumps(result))
"""


def run_tests(
    code: str, tests: list[str], imports: list[str], timeout: float = 5.0
) -> dict:
    started = perf_counter()
    try:
        check_code_policy(code)
    except (ValueError, SyntaxError) as error:
        return {
            "status": "invalid_code",
            "passed": False,
            "feedback": str(error)[:500],
            "wall_s": perf_counter() - started,
        }
    if not Path("/usr/bin/sandbox-exec").exists():
        raise RuntimeError(
            "This pilot requires macOS sandbox-exec; no unisolated fallback"
        )
    try:
        process = subprocess.run(
            [
                "/usr/bin/sandbox-exec",
                "-p",
                sandbox_profile(),
                str(Path(sys.executable).resolve()),
                "-I",
                "-S",
                "-c",
                RUNNER,
            ],
            input=json.dumps({"code": code, "tests": tests, "imports": imports}),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            cwd="/",
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "passed": False,
            "feedback": "Test execution timed out",
            "wall_s": perf_counter() - started,
        }
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        result = {
            "status": "resource_limit"
            if process.returncode in {-signal.SIGKILL, -signal.SIGXCPU}
            else "tool_error",
            "passed": False,
            "feedback": f"Isolated test process exited {process.returncode}: {process.stderr[-500:]}",
        }
    result["wall_s"] = perf_counter() - started
    return result


def task_messages(task: dict) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": task["prompt"]
            + "\n\nYour function must pass this example:\n"
            + task["test_list"][0],
        },
    ]


def repair_feedback(result: dict) -> str:
    return (
        "The visible test failed.\n"
        + result.get("feedback", result["status"])
        + "\n"
        + result.get("traceback", "")
        + "\nReturn a complete corrected Python solution."
    )


def repeat_fraction(tokens: list[int], previous: list[list[int]], n: int = 4) -> float:
    """Fraction of current code tokens covered by historical matching n-grams."""
    history = {
        tuple(seq[i : i + n]) for seq in previous for i in range(len(seq) - n + 1)
    }
    covered = set()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i : i + n]) in history:
            covered.update(range(i, i + n))
    return len(covered) / len(tokens) if tokens else 0.0
