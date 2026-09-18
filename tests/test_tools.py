import json
import time

import pytest

from agent_harness.domain import (
    ApprovalRequired, CallConflictError, LeaseLostError, RiskLevel, UnsafeRetryError,
)
from agent_harness.store import SQLiteStore
from agent_harness.tools import ToolGateway, ToolRegistry, ToolSpec


def setup_gateway(tmp_path, risk, handler):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("test tools")
    registry = ToolRegistry()
    registry.register(ToolSpec("tool", "test tool", risk, handler=handler))
    return store, workflow_id, ToolGateway(store, registry)


def test_success_is_replayed_without_duplicate_side_effect(tmp_path):
    effects = []
    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.IDEMPOTENT_WRITE, lambda args: effects.append(args) or {"ok": True}
    )
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"x": 1},
                  purpose="test", call_id="call-1")

    assert gateway.execute(**kwargs) == {"ok": True}
    assert gateway.execute(**kwargs) == {"ok": True}
    assert effects == [{"x": 1}]


def test_non_idempotent_call_requires_approval_before_side_effect(tmp_path):
    effects = []
    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.NON_IDEMPOTENT_WRITE,
        lambda args: effects.append(args) or {"sent": True},
    )
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"text": "hello"},
                  purpose="notify customer", call_id="call-send")

    with pytest.raises(ApprovalRequired) as caught:
        gateway.execute(**kwargs)
    assert effects == []

    store.decide_approval(caught.value.approval_id, True, "reviewer")
    assert gateway.execute(**kwargs) == {"sent": True}
    assert len(effects) == 1


def test_call_id_cannot_be_reused_with_different_arguments(tmp_path):
    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.READ_ONLY, lambda args: args
    )
    common = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", purpose="read", call_id="same")
    gateway.execute(arguments={"x": 1}, **common)
    with pytest.raises(CallConflictError):
        gateway.execute(arguments={"x": 2}, **common)


def test_atomic_claim_allows_only_one_running_owner(tmp_path):
    store, workflow_id, gateway = setup_gateway(tmp_path, RiskLevel.READ_ONLY, lambda args: args)
    fingerprint = gateway.fingerprint({"x": 1})
    with store.transaction() as conn:
        now = "2026-01-01T00:00:00+00:00"
        conn.execute(
            """INSERT INTO tool_calls
            (call_id, session_id, workflow_id, task_id, tool_name, risk, args_fingerprint,
             args_json, purpose, status, attempt, lease_owner, lease_until, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 1, 'other', 9999999999, ?, ?)""",
            ("busy", store.session_id(workflow_id), workflow_id, "task", "tool",
             RiskLevel.READ_ONLY, fingerprint, '{"x":1}', "read", now, now),
        )
    with pytest.raises(RuntimeError, match="already running"):
        gateway.execute(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                        task_id="task", tool_name="tool", arguments={"x": 1},
                        purpose="read", call_id="busy")


def test_non_idempotent_failure_becomes_uncertain_and_is_not_retried(tmp_path):
    effects = []

    def ambiguous_failure(args):
        effects.append(args)
        raise TimeoutError("response lost after send")

    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.NON_IDEMPOTENT_WRITE, ambiguous_failure
    )
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"text": "hello"},
                  purpose="notify", call_id="ambiguous")
    with pytest.raises(ApprovalRequired) as caught:
        gateway.execute(**kwargs)
    store.decide_approval(caught.value.approval_id, True, "reviewer")
    with pytest.raises(TimeoutError):
        gateway.execute(**kwargs)
    with pytest.raises(UnsafeRetryError):
        gateway.execute(**kwargs)
    assert len(effects) == 1


def test_late_worker_cannot_clobber_the_result_of_the_new_lease_holder(tmp_path):
    """A slow handler that outlives its lease must not overwrite the new owner."""
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("lease handover")
    registry = ToolRegistry()
    calls: list[int] = []
    winner: dict[str, object] = {}

    def handler(args):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(0.05)  # the first worker's 0.01s lease expires here
            winner["result"] = ToolGateway(
                store, registry, worker_id="worker-b"
            ).execute(**kwargs)
            return {"invocation": 1}
        return {"invocation": 2}

    registry.register(ToolSpec("tool", "test tool", RiskLevel.IDEMPOTENT_WRITE, handler=handler))
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"x": 1},
                  purpose="test", call_id="handover")
    gateway_a = ToolGateway(store, registry, worker_id="worker-a", lease_seconds=0.01)

    assert gateway_a.execute(**kwargs) == {"invocation": 2}
    assert winner["result"] == {"invocation": 2}
    with store.transaction() as conn:
        row = conn.execute("SELECT status, result_json FROM tool_calls WHERE call_id='handover'").fetchone()
    assert row["status"] == "succeeded"
    assert json.loads(row["result_json"]) == {"invocation": 2}
    events = [e["event_type"] for e in store.recent_events(workflow_id, 50)]
    assert "tool.lease_lost" in events


def test_lost_lease_on_non_idempotent_call_is_surfaced_not_overwritten(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("non-idempotent handover")
    registry = ToolRegistry()

    def handler(args):
        # An operator reconciled the call elsewhere while we were still running.
        with store.transaction(immediate=True) as conn:
            conn.execute("UPDATE tool_calls SET lease_owner='operator' WHERE call_id='send'")
        return {"sent": True}

    registry.register(ToolSpec("tool", "test tool", RiskLevel.NON_IDEMPOTENT_WRITE, handler=handler))
    gateway = ToolGateway(store, registry, worker_id="worker-a")
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"text": "hi"},
                  purpose="notify", call_id="send")

    with pytest.raises(ApprovalRequired) as caught:
        gateway.execute(**kwargs)
    store.decide_approval(caught.value.approval_id, True, "reviewer")

    with pytest.raises(LeaseLostError):
        gateway.execute(**kwargs)
    with store.transaction() as conn:
        row = conn.execute("SELECT status FROM tool_calls WHERE call_id='send'").fetchone()
    assert row["status"] == "running"  # untouched by the worker that lost the lease
