import pytest

from tiny_llm.agent_pilot import (
    check_code_policy,
    extract_code,
    repair_feedback,
    repeat_fraction,
    run_tests,
    task_messages,
)


def test_prompt_excludes_reference_and_heldout_tests():
    task = {
        "prompt": "Solve this.",
        "test_list": ["assert f(1)==1", "assert f(2)==2"],
        "code": "SECRET_REFERENCE",
    }
    content = str(task_messages(task))
    assert "assert f(1)==1" in content
    assert "assert f(2)==2" not in content
    assert "SECRET_REFERENCE" not in content


def test_code_extraction_and_policy():
    assert (
        extract_code("Here is code:\n```python\ndef f(): return 1\n```\nDone.")
        == "def f(): return 1"
    )
    with pytest.raises(ValueError):
        check_code_policy("import os\nos.system('echo bad')")
    with pytest.raises(ValueError):
        check_code_policy("open('/etc/passwd').read()")


def test_isolated_tool_pass_fail_and_timeout():
    assert run_tests("def f(): return 1", ["assert f() == 1"], [])["passed"]
    failed = run_tests("def f(): return 2", ["assert f() == 1"], [])
    assert failed["status"] == "failed"
    assert "AssertionError" in repair_feedback(failed)
    assert "returned 2; expected 1" in failed["feedback"]
    assert run_tests("while True: pass", [], [], timeout=0.25)["status"] == "timeout"


def test_repeated_spans_do_not_cross_history_boundaries():
    assert repeat_fraction([1, 2, 3, 4, 9], [[1, 2, 3, 4]]) == 0.8
    assert repeat_fraction([1, 2, 3, 4], [[1, 2], [3, 4]]) == 0.0
    assert repeat_fraction([], []) == 0.0


def test_test_tool_restricts_home_reads_and_file_writes():
    probe = "try:\n    open('/private/tmp/agent-pilot-write-probe', 'w')\nexcept PermissionError:\n    pass\nelse:\n    raise AssertionError('write was permitted')"
    assert run_tests("def f(): return 1", [probe], [])["passed"]
    home_probe = (
        "try:\n    open("
        + repr(str(__import__("pathlib").Path(__file__).resolve()))
        + ").read()\nexcept PermissionError:\n    pass\nelse:\n    raise AssertionError('home read was permitted')"
    )
    assert run_tests("def f(): return 1", [home_probe], [])["passed"]
