import time

from agent_harness.agents import DemoAgent
from agent_harness.domain import Task, TaskStatus, WorkflowStatus
from agent_harness.orchestrator import Orchestrator
from agent_harness.store import SQLiteStore


def build_workflow(store: SQLiteStore) -> str:
    workflow_id = store.create_workflow("resume after crash")
    store.add_task(Task("analysis", workflow_id, "analyze", "analyst", "analyze"))
    store.add_task(Task("build", workflow_id, "build", "backend", "build", depends_on=["analysis"]))
    return workflow_id


def test_task_abandoned_by_a_dead_worker_is_reclaimed_and_completed(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = build_workflow(store)
    # A worker claimed "analysis" and then the process died mid-run.
    assert store.start_task("analysis", worker_id="dead-worker", lease_seconds=0.01)
    time.sleep(0.02)

    orchestrator = Orchestrator(store, {
        "analyst": DemoAgent(), "backend": DemoAgent(), "tester": DemoAgent(),
    })
    assert orchestrator.run(workflow_id) == WorkflowStatus.SUCCEEDED
    assert all(task.status == TaskStatus.SUCCEEDED for task in store.list_tasks(workflow_id))
    reclaimed = [e for e in store.recent_events(workflow_id, 100) if e["event_type"] == "task.reclaimed"]
    assert len(reclaimed) == 1
    assert reclaimed[0]["task_id"] == "analysis"


def test_live_lease_is_not_stolen_from_a_running_worker(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = build_workflow(store)
    assert store.start_task("analysis", worker_id="alive-worker", lease_seconds=300)

    assert store.reclaim_expired_tasks(workflow_id) == []
    assert store.get_task("analysis").status == TaskStatus.RUNNING

    # A second orchestrator must report work in progress, not silently finish.
    other = Orchestrator(store, {"analyst": DemoAgent(), "backend": DemoAgent(), "tester": DemoAgent()})
    assert other.run(workflow_id) == WorkflowStatus.RUNNING


def test_lease_is_released_on_every_terminal_transition(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("lease hygiene")
    for task_id in ("done", "failed", "waiting"):
        store.add_task(Task(task_id, workflow_id, task_id, "analyst", "work"))

    for task_id in ("done", "failed", "waiting"):
        assert store.start_task(task_id, worker_id="w", lease_seconds=300)

    store.finish_task("done", {"ok": True})
    store.fail_task("failed", "boom", retryable=False)
    store.wait_task_for_approval("waiting")

    for task_id in ("done", "failed", "waiting"):
        task = store.get_task(task_id)
        assert task.status != TaskStatus.RUNNING, task_id
    # Nothing is reclaimable, because no task holds a stale lease any more.
    assert store.reclaim_expired_tasks(workflow_id) == []
