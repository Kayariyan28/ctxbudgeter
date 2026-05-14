"""Test the pytest plugin itself — the `ctxbudgeter_golden` fixture and update flag.

Uses pytester to spawn a sub-pytest session.
"""

from __future__ import annotations

pytest_plugins = ["pytester"]


def test_plugin_fixture_works(pytester) -> None:
    pytester.makepyfile(
        """
        from ctxbudgeter import ContextPack

        def test_golden(ctxbudgeter_golden):
            pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
            pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
            pack.add(name="task", content="t", kind="task", required=True)
            ctxbudgeter_golden().check(pack.compile())
        """
    )
    # First run: creates golden file
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=1)

    # Second run: should still pass (matches golden)
    result2 = pytester.runpytest("-v")
    result2.assert_outcomes(passed=1)


def test_plugin_update_flag(pytester) -> None:
    pytester.makepyfile(
        """
        from ctxbudgeter import ContextPack

        def test_golden(ctxbudgeter_golden):
            pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
            pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
            pack.add(name="task", content="task body", required=True)
            ctxbudgeter_golden().check(pack.compile())
        """
    )
    # First run: creates golden
    pytester.runpytest("-v").assert_outcomes(passed=1)
    # Modify the test to add a new item — should fail
    pytester.makepyfile(
        """
        from ctxbudgeter import ContextPack

        def test_golden(ctxbudgeter_golden):
            pack = ContextPack(token_budget=5_000, reserved_output_tokens=500)
            pack.add(name="sys", content="rules", kind="system", required=True, cache_policy="stable")
            pack.add(name="task", content="task body", required=True)
            pack.add(name="extra", content="new item", priority=50)
            ctxbudgeter_golden().check(pack.compile())
        """
    )
    pytester.runpytest("-v").assert_outcomes(failed=1)
    # Update — should refresh golden
    pytester.runpytest("-v", "--ctxbudgeter-update-golden").assert_outcomes(passed=1)
    # Now it should pass without the flag
    pytester.runpytest("-v").assert_outcomes(passed=1)
