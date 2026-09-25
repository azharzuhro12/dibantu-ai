"""Tests for the Step 6 evaluation framework.

Everything runs without the GLM API: the evaluator only ever talks
to a ScriptedGLMClient while the real tools run against the seeded
PostgreSQL test database (no API key needed; see conftest.py).
"""

from __future__ import annotations

import pytest

from app.tools.registry import get_tool_schemas
from evaluation import CASES, CATEGORIES, EvalCase, load_cases
from evaluation.dataset import GLMTurn, say, tools
from evaluation.evaluator import (
    GLMScriptError,
    format_report,
    main,
    run_case,
    run_evaluation,
)


@pytest.fixture(autouse=True)
def _business_database(test_database):
    """The evaluator runs the real tools against the PostgreSQL test DB."""
    yield


def case_by_id(case_id: str) -> EvalCase:
    return next(case for case in CASES if case.id == case_id)


# --- dataset shape ----------------------------------------------------------


def test_dataset_has_at_least_30_cases() -> None:
    assert len(load_cases()) >= 30


def test_dataset_covers_every_category() -> None:
    present = {case.category for case in CASES}
    assert present == set(CATEGORIES)


def test_all_cases_have_required_fields() -> None:
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    # Advertised = dispatchable + sensitive-request schemas: everything
    # the scripted GLM may legitimately ask for (sensitive calls get
    # intercepted into approvals instead of dispatched).
    advertised = {schema["name"] for schema in get_tool_schemas()}
    for case in CASES:
        assert case.id
        assert case.user_message.strip()
        assert case.expected_behavior.strip()
        assert case.category in CATEGORIES
        assert case.glm_turns, f"{case.id}: needs at least one scripted turn"
        assert case.glm_turns[-1].text, f"{case.id}: last turn must be final text"
        assert all(
            turn.tool_calls or turn.text for turn in case.glm_turns
        ), f"{case.id}: every turn needs tool calls or text"
        minimum, maximum = case.expected_tool_count
        assert 0 <= minimum <= maximum <= 5, f"{case.id}: bad tool count range"
        scripted = [call.name for turn in case.glm_turns for call in turn.tool_calls]
        assert minimum <= len(scripted) <= maximum, (
            f"{case.id}: script has {len(scripted)} calls, outside "
            f"{case.expected_tool_count}"
        )
        if not case.expects_tool_error:
            assert set(case.expected_tools) <= advertised, (
                f"{case.id}: expected tools must be advertised to the model"
            )
        assert set(case.expected_tools) == set(scripted), (
            f"{case.id}: expected_tools must match the scripted trajectory"
        )
        assert all(fact.strip() for fact in case.grounding_facts)
        assert case.expected_state.monthly_orders >= 0


# --- evaluator behaviour ----------------------------------------------------


def test_evaluator_identifies_a_passing_case() -> None:
    result = run_case(case_by_id("ev_001"))

    assert result.passed is True
    assert result.failures == ()
    assert result.tools_executed == ("check_stock",)
    assert "24" in result.reply


def test_evaluator_identifies_failed_tool_selection() -> None:
    wrong_tool = EvalCase(
        id="ev_wrong_tool",
        category="stock_check",
        user_message="Berapa stok kopi susu?",
        expected_behavior="Seharusnya update stok (kesalahan disengaja).",
        expected_tools=("update_stock",),
        expected_tool_count=(1, 1),
        glm_turns=(
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            say("Stok Kopi Susu 24 unit."),
        ),
        grounding_facts=("24",),
    )

    result = run_case(wrong_tool)

    assert result.passed is False
    assert result.tool_selection_ok is False
    assert result.tools_executed == ("check_stock",)
    assert any("tool selection" in failure for failure in result.failures)


def test_evaluator_handles_multi_step_cases() -> None:
    result = run_case(case_by_id("ev_033"))

    assert result.passed is True
    assert result.tools_executed == ("search_customer", "check_stock", "create_order")
    assert result.tool_count_ok is True
    assert result.completion_ok is True


def test_evaluator_handles_tool_errors_safely() -> None:
    error_cases = [case for case in CASES if case.expects_tool_error]

    assert error_cases, "dataset must contain tool_error cases"
    for case in error_cases:
        result = run_case(case)
        assert result.error_handling_ok is True, case.id
        assert result.passed is True, (case.id, result.failures)
        assert result.completion_ok is True  # store untouched despite the error


def test_evaluator_reports_exhausted_script() -> None:
    """A trajectory that never yields text fails the case, not the runner."""
    runaway = EvalCase(
        id="ev_runaway",
        category="stock_check",
        user_message="Cek terus ya.",
        expected_behavior="Script tidak pernah selesai.",
        expected_tools=("check_stock",),
        expected_tool_count=(1, 1),
        glm_turns=(tools(("check_stock", {"product_name": "Kopi Susu"})),),
    )

    result = run_case(runaway)

    assert result.passed is False
    assert any("GLMScriptError" in failure for failure in result.failures)


def test_metrics_are_calculated_correctly() -> None:
    passing = case_by_id("ev_001")
    failing = EvalCase(
        id="ev_wrong_tool",
        category="stock_check",
        user_message="Berapa stok teh manis?",
        expected_behavior="Kesalahan pemilihan tool disengaja.",
        expected_tools=("get_sales_report",),
        expected_tool_count=(1, 1),
        glm_turns=(
            tools(("check_stock", {"product_name": "Teh Manis"})),
            say("Stok Teh Manis 30 unit."),
        ),
        grounding_facts=("30",),
    )

    report = run_evaluation([passing, failing])

    assert report.total == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.pass_rate == 50.0
    assert report.tool_selection_accuracy == 50.0
    assert report.tool_count_accuracy == 100.0  # both ran one tool
    assert report.task_completion_rate == 100.0  # states are as expected
    assert report.response_grounding_rate == 100.0  # replies cite real data
    assert report.tool_error_handling_rate is None  # no error cases


def test_full_dataset_passes_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole evaluation is deterministic and needs no API key."""
    monkeypatch.setenv("GLM_API_KEY", "")

    report = run_evaluation()

    assert report.total == len(CASES)
    assert report.failed == 0, [r.case_id for r in report.results if not r.passed]
    assert report.pass_rate == 100.0
    assert report.tool_error_handling_rate == 100.0


def test_evaluation_does_not_require_a_real_glm_client() -> None:
    """ScriptedGLMClient answers tool turns and final text by itself."""
    from evaluation.evaluator import ScriptedGLMClient

    client = ScriptedGLMClient(
        [
            tools(("check_stock", {"product_name": "Kopi Susu"})),
            say("Stok Kopi Susu 24 unit."),
        ]
    )

    import asyncio

    first = asyncio.run(
        client.complete_with_tools(
            [{"role": "user", "content": "Berapa stok kopi susu?"}],
            system="test",
            tools=[],
        )
    )
    assert first.tool_name == "check_stock"
    second = asyncio.run(
        client.complete_with_tools(
            [
                {"role": "user", "content": "Berapa stok kopi susu?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "i", "name": "check_stock", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "i", "content": '{"stock": 24}'}]},
            ],
            system="test",
            tools=[],
        )
    )
    assert second.text == "Stok Kopi Susu 24 unit."
    assert client.executed_tools == ["check_stock"]
    assert '{"stock": 24}' in client.tool_results
    with pytest.raises(GLMScriptError):
        asyncio.run(
            client.complete_with_tools([], system="test", tools=[])
        )


def test_cli_prints_computed_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    report = run_evaluation()

    main()

    out = capsys.readouterr().out
    assert "DibantuAI Agent Evaluation" in out
    assert f"Cases: {report.total}" in out
    assert f"Passed: {report.passed}" in out
    assert f"Failed: {report.failed}" in out
    assert f"Pass Rate: {report.pass_rate}%" in out
    assert (
        f"Tool Error Handling Rate: {report.tool_error_handling_rate}%" in out
    )
    # Failed-case details only appear when something actually failed.
    assert ("Failed cases:" in out) == (report.failed > 0)
