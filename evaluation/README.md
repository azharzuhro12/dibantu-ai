# DibantuAI Agent Evaluation (Step 6 → Step 16)

A lightweight, deterministic, offline evaluation framework for the
DibantuAI agent. No LangChain/LangGraph, no external model API, no API
key, no network — every run produces identical results.

## Structure

- `dataset.py` — the evaluation cases. Each case has an `id`,
  `category`, `user_message`, `expected_behavior`, `expected_tools`,
  `expected_tool_count` (inclusive min/max), a scripted GLM trajectory
  (`glm_turns`), `grounding_facts`, an `expected_state`, and
  `expects_tool_error`. Step 14 added seeded `memories` rows; Step 16
  added an optional `approval_flow` (a scripted human lifecycle applied
  after the agent turn).
- `evaluator.py` — `ScriptedGLMClient`, `run_case`, `run_evaluation`,
  metrics, and the CLI (`python -m evaluation.evaluator`).

## How a case runs

The scripted GLM turns are replayed through the **real** `Agent` loop
(`app.agent.agent`) while the **real** registry business tools
(`app.tools.registry` → `app.tools.business_tools`) execute against a
freshly re-seeded PostgreSQL store (`reset_mock_data()` before every
case). The script decides which tools GLM *requests*; the evaluator
verifies what the agent actually *did*.

Every case reset is a TRUNCATE, so the CLI (`python -m
evaluation.evaluator`) and pytest both pin the run to an isolated
scratch database first (`db_isolation.py` → `dibantu_ai_test`); the
main business database behind `DATABASE_URL`/`.env` is never touched.
The scratch server is the Docker Compose Postgres (start it with
`docker compose up -d postgres`); when it is unreachable the evaluator
exits with a clear message instead of resetting anything.

Since Step 12, `knowledge_*` cases also run the **real**
`search_knowledge_base` tool — against a scratch vector store the
evaluator ingests into a temporary directory with deterministic
`hashing` embeddings (no model download, no network; the developer's
real `data/rag` store is never touched).

Since Step 16, `approval_*` cases drive the **real** approval
lifecycle after the agent turn: the scripted agent request creates (or
must not create) a pending approval, then `ApprovalFlow.steps` apply
the human side deterministically through the real manager and executor
(`approve`, `reject`, `execute`, `execute_conflict`,
`verify_persisted` — a dispose/recreate engine cycle that simulates a
restart) against the same scratch database. `observability_trace`
cases verify the executor's trace contract: exactly one run with
`APPROVAL_EXECUTING` followed by the terminal event, and metadata
never carrying the payload. The CLI report separates five groups —
Business (Step 6), Knowledge/RAG (Step 12), Memory (Step 14), Approval
execution (Step 16), and Observability — so an addition in one area
can never silently shift another group's picture.

## Checks per case (all must hold for a pass)

| Check | Meaning |
|---|---|
| Tool selection | The set of executed tools equals `expected_tools`. |
| Tool call count | Executions within `expected_tool_count` bounds. |
| Task completion | Final mock-store state matches `expected_state` (product stocks + 30-day order count, read via public tools); for Step 16 cases this also includes the scripted approval lifecycle outcome (terminal status, execution result, trace contract). |
| Tool error handling | For error cases: the failed/unknown tool came back as an `{"error": ...}` tool_result and the run completed. |
| Response grounding | Every `grounding_fact` appears in the final reply AND in a tool result the agent really received. |

Scripted final replies must never state numbers the tools did not
return — that constraint is what makes the grounding check meaningful.

## Metrics

Total / passed / failed cases, pass rate, tool selection accuracy,
tool call count accuracy, task completion rate, tool error handling
rate (over error cases only; `n/a` when there are none), and response
grounding rate. All are plain per-check rates — there is deliberately
**no weighted overall score**.

## Running

```bash
python -m evaluation.evaluator   # CLI report
pytest -q tests/test_evaluation.py
```

## Adding a case

Append a `case(...)` entry to `CASES` in `dataset.py`. Give it a unique
id, one of the categories in `CATEGORIES`, a scripted trajectory whose
tool inputs match the seeded store (see the seed reference in the
module docstring), and expectations derived only from what the tools
actually return. Sensitive-action cases script the tool call the model
requests (e.g. `refund_order`), keep `expected_state` at the
pre-execution values, and describe the human side via
`approval_flow=ApprovalFlow(...)`; the state after execution is then
asserted through the same public `expected_state` check.
