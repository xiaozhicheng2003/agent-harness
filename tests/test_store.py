import sqlite3

from agent_harness.domain import Task
from agent_harness.store import SQLiteStore


def test_store_closes_every_connection_it_opens(tmp_path, monkeypatch):
    """sqlite3's context manager commits but does not close; each op must close."""
    opened: list[sqlite3.Connection] = []
    closed: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def close(self) -> None:
            closed.append(self)
            super().close()

    def tracking_connect(*args, **kwargs):
        kwargs["factory"] = TrackedConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("no leaks")
    store.add_task(Task("t", workflow_id, "t", "analyst", "inspect"))
    for _ in range(10):
        store.get_workflow(workflow_id)
        store.get_task("t")
        store.list_tasks(workflow_id)
        store.list_artifacts(workflow_id)
        store.recent_events(workflow_id)
        store.list_approvals(workflow_id)

    # Strong references are held above, so nothing here can be reclaimed by gc.
    assert len(opened) > 50
    assert len(opened) == len(closed)
