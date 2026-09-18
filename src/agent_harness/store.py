from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .domain import Task, TaskStatus, WorkflowStatus


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class SQLiteStore:
    """Durable execution state and append-only audit events."""

    def __init__(self, path: str | Path = "harness.db") -> None:
        self.path = str(path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection that is always closed on exit.

        ``sqlite3.Connection`` doubles as a transaction context manager, so
        ``with sqlite3.connect(...) as conn`` commits but never closes. Using it
        as a connection scope leaks one file descriptor per call, which exhausts
        the default ``ulimit -n`` after a few hundred reads.
        """
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _initialize(self) -> None:
        with self._connect() as conn:
            # WAL is a persistent database property; setting it on every
            # connection would needlessly take a write lock per connect.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL REFERENCES workflows(id),
                    title TEXT NOT NULL,
                    role TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    status TEXT NOT NULL,
                    depends_on TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 2,
                    lease_owner TEXT,
                    lease_until REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_workflow_status
                    ON tasks(workflow_id, status);
                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    args_fingerprint TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    lease_owner TEXT,
                    lease_until REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    call_id TEXT NOT NULL UNIQUE,
                    tool_name TEXT NOT NULL,
                    args_fingerprint TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT,
                    note TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    digest TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    workflow_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    through_event_seq INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(workflow_id, scope, version)
                );
                CREATE TABLE IF NOT EXISTS agent_steps (
                    task_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    turn_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, step)
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(namespace, memory_key, source_task_id)
                );
                """
            )
            self._migrate(conn)

    @staticmethod
    def _add_missing_columns(
        conn: sqlite3.Connection, table: str, columns: dict[str, str],
    ) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additive migrations so databases from an earlier version keep working."""
        self._add_missing_columns(conn, "tasks", {
            "lease_owner": "TEXT",
            "lease_until": "REAL",
        })

    def create_workflow(self, goal: str, *, workflow_id: str | None = None) -> str:
        workflow_id = workflow_id or f"wf_{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO workflows VALUES (?, ?, ?, ?, ?, ?)",
                (workflow_id, goal, WorkflowStatus.PENDING, f"session_{uuid.uuid4().hex}", now, now),
            )
            self._append_event(conn, workflow_id, None, "workflow.created", {"goal": goal})
        return workflow_id

    def add_task(self, task: Task) -> None:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO tasks
                (id, workflow_id, title, role, instructions, status, depends_on,
                 input_json, output_json, error, attempts, max_attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.id, task.workflow_id, task.title, task.role, task.instructions,
                    task.status, dumps(task.depends_on), dumps(task.input), None, None,
                    task.attempts, task.max_attempts, now, now,
                ),
            )
            self._append_event(conn, task.workflow_id, task.id, "task.created", {"role": task.role})

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM workflows WHERE id=?", (workflow_id,)).fetchone()
        if row is None:
            raise KeyError(workflow_id)
        return dict(row)

    def session_id(self, workflow_id: str) -> str:
        return str(self.get_workflow(workflow_id)["session_id"])

    def set_workflow_status(self, workflow_id: str, status: WorkflowStatus) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE workflows SET status=?, updated_at=? WHERE id=?",
                (status, utc_now(), workflow_id),
            )
            self._append_event(conn, workflow_id, None, "workflow.status", {"status": status})

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"], workflow_id=row["workflow_id"], title=row["title"],
            role=row["role"], instructions=row["instructions"], status=TaskStatus(row["status"]),
            depends_on=json.loads(row["depends_on"]), input=json.loads(row["input_json"]),
            output=json.loads(row["output_json"]) if row["output_json"] else None,
            error=row["error"], attempts=row["attempts"], max_attempts=row["max_attempts"],
        )

    def get_task(self, task_id: str) -> Task:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        return self._row_to_task(row)

    def list_tasks(self, workflow_id: str) -> list[Task]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE workflow_id=? ORDER BY created_at, id", (workflow_id,)
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def ready_tasks(self, workflow_id: str) -> list[Task]:
        tasks = self.list_tasks(workflow_id)
        succeeded = {task.id for task in tasks if task.status == TaskStatus.SUCCEEDED}
        return [
            task for task in tasks
            if task.status == TaskStatus.PENDING and set(task.depends_on).issubset(succeeded)
        ]

    def start_task(
        self,
        task_id: str,
        *,
        worker_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> bool:
        """Atomically claim a pending task under a lease.

        The lease is what makes a crash recoverable: a task left ``running`` by a
        process that died is reclaimed by :meth:`reclaim_expired_tasks` instead of
        blocking the workflow forever.
        """
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                """UPDATE tasks SET status=?, attempts=attempts+1, lease_owner=?,
                   lease_until=?, updated_at=?
                   WHERE id=? AND status=?""",
                (TaskStatus.RUNNING, worker_id, time.time() + lease_seconds, utc_now(),
                 task_id, TaskStatus.PENDING),
            )
            if cursor.rowcount:
                row = conn.execute("SELECT workflow_id FROM tasks WHERE id=?", (task_id,)).fetchone()
                self._append_event(conn, row[0], task_id, "task.started", {"worker_id": worker_id})
            return bool(cursor.rowcount)

    def reclaim_expired_tasks(self, workflow_id: str, *, now: float | None = None) -> list[str]:
        """Return tasks abandoned by a dead worker to the pending pool.

        Re-running a reclaimed task is safe because the tool ledger replays calls
        that already succeeded and refuses to blindly retry non-idempotent ones.
        """
        deadline = time.time() if now is None else now
        with self.transaction(immediate=True) as conn:
            rows = conn.execute(
                """SELECT id, lease_owner FROM tasks
                   WHERE workflow_id=? AND status=?
                     AND (lease_until IS NULL OR lease_until <= ?)""",
                (workflow_id, TaskStatus.RUNNING, deadline),
            ).fetchall()
            for row in rows:
                conn.execute(
                    """UPDATE tasks SET status=?, lease_owner=NULL, lease_until=NULL, updated_at=?
                       WHERE id=?""",
                    (TaskStatus.PENDING, utc_now(), row["id"]),
                )
                self._append_event(conn, workflow_id, row["id"], "task.reclaimed", {
                    "previous_owner": row["lease_owner"],
                })
            return [row["id"] for row in rows]

    def finish_task(self, task_id: str, output: dict[str, Any]) -> None:
        with self.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT workflow_id, status, output_json FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            encoded = dumps(output)
            if row["status"] == TaskStatus.SUCCEEDED:
                if row["output_json"] == encoded:
                    return
                raise ValueError(f"task {task_id} already succeeded with different output")
            conn.execute(
                """UPDATE tasks SET status=?, output_json=?, error=NULL, lease_owner=NULL,
                   lease_until=NULL, updated_at=? WHERE id=?""",
                (TaskStatus.SUCCEEDED, encoded, utc_now(), task_id),
            )
            self._append_event(conn, row["workflow_id"], task_id, "task.succeeded", {"output": output})

    def fail_task(self, task_id: str, error: str, *, retryable: bool) -> None:
        with self.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT workflow_id, attempts, max_attempts FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            status = TaskStatus.PENDING if retryable and row["attempts"] < row["max_attempts"] else TaskStatus.FAILED
            conn.execute(
                """UPDATE tasks SET status=?, error=?, lease_owner=NULL, lease_until=NULL,
                   updated_at=? WHERE id=?""",
                (status, error, utc_now(), task_id),
            )
            self._append_event(conn, row["workflow_id"], task_id, "task.failed", {"error": error, "retryable": retryable})

    def wait_task_for_approval(self, task_id: str) -> None:
        with self.transaction(immediate=True) as conn:
            row = conn.execute("SELECT workflow_id, status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] == TaskStatus.WAITING_APPROVAL:
                return
            conn.execute(
                """UPDATE tasks SET status=?, lease_owner=NULL, lease_until=NULL, updated_at=?
                   WHERE id=?""",
                (TaskStatus.WAITING_APPROVAL, utc_now(), task_id),
            )
            self._append_event(conn, row["workflow_id"], task_id, "task.waiting_approval", {})

    def resume_approved_tasks(self, workflow_id: str) -> int:
        with self.transaction(immediate=True) as conn:
            cursor = conn.execute(
                """UPDATE tasks SET status=?, lease_owner=NULL, lease_until=NULL, updated_at=?
                   WHERE workflow_id=? AND status=? AND NOT EXISTS (
                     SELECT 1 FROM approvals a
                     WHERE a.task_id=tasks.id AND a.status!='approved'
                   )""",
                (TaskStatus.PENDING, utc_now(), workflow_id, TaskStatus.WAITING_APPROVAL),
            )
            if cursor.rowcount:
                self._append_event(
                    conn, workflow_id, None, "task.resumed_after_approval",
                    {"count": cursor.rowcount},
                )
            return cursor.rowcount

    def add_artifact(self, workflow_id: str, task_id: str, artifact: dict[str, Any]) -> str:
        artifact_id = artifact.get("id") or f"artifact_{uuid.uuid4().hex[:12]}"
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (artifact_id, workflow_id, task_id, artifact["name"], artifact["uri"],
                 artifact.get("kind", "file"), artifact.get("digest"),
                 dumps(artifact.get("metadata", {})), utc_now()),
            )
            self._append_event(conn, workflow_id, task_id, "artifact.created", {"id": artifact_id, **artifact})
        return artifact_id

    def list_artifacts(self, workflow_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE workflow_id=? ORDER BY created_at", (workflow_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def append_event(self, workflow_id: str, task_id: str | None, event_type: str, payload: Any) -> None:
        with self.transaction() as conn:
            self._append_event(conn, workflow_id, task_id, event_type, payload)

    def _append_event(
        self, conn: sqlite3.Connection, workflow_id: str, task_id: str | None,
        event_type: str, payload: Any,
    ) -> None:
        conn.execute(
            "INSERT INTO events(workflow_id, task_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (workflow_id, task_id, event_type, dumps(payload), utc_now()),
        )

    def recent_events(self, workflow_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE workflow_id=? ORDER BY seq DESC LIMIT ?",
                (workflow_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def save_summary(self, workflow_id: str, scope: str, content: str, through_seq: int) -> None:
        with self.transaction() as conn:
            version = conn.execute(
                "SELECT COALESCE(MAX(version), 0)+1 FROM summaries WHERE workflow_id=? AND scope=?",
                (workflow_id, scope),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO summaries VALUES (?, ?, ?, ?, ?, ?)",
                (workflow_id, scope, version, through_seq, content, utc_now()),
            )

    def latest_summary(self, workflow_id: str, scope: str = "workflow") -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM summaries WHERE workflow_id=? AND scope=?
                   ORDER BY version DESC LIMIT 1""",
                (workflow_id, scope),
            ).fetchone()
        return dict(row) if row else None

    def save_agent_step(self, task_id: str, step: int, turn: dict[str, Any]) -> None:
        """Persist a model decision once; recovery reuses it instead of asking again."""
        with self.transaction(immediate=True) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO agent_steps VALUES (?, ?, ?, ?)",
                (task_id, step, dumps(turn), utc_now()),
            )

    def get_agent_step(self, task_id: str, step: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT turn_json FROM agent_steps WHERE task_id=? AND step=?", (task_id, step)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put_memory(
        self,
        *,
        namespace: str,
        memory_key: str,
        content: str,
        source_task_id: str,
        confidence: float,
    ) -> str:
        digest = uuid.uuid5(
            uuid.NAMESPACE_URL, f"{namespace}:{memory_key}:{source_task_id}"
        ).hex[:16]
        memory_id = f"memory_{digest}"
        with self.transaction(immediate=True) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO memories
                (id, namespace, memory_key, content, source_task_id, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (memory_id, namespace, memory_key, content, source_task_id, confidence, utc_now()),
            )
        return memory_id

    def list_memories(self, namespace: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE namespace=? ORDER BY created_at", (namespace,)
            ).fetchall()
        return [dict(row) for row in rows]

    def decide_approval(self, approval_id: str, approved: bool, actor: str, note: str = "") -> None:
        with self.transaction(immediate=True) as conn:
            row = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
            if row is None:
                raise KeyError(approval_id)
            if row["status"] != "pending":
                raise ValueError(f"approval already decided: {row['status']}")
            status = "approved" if approved else "rejected"
            conn.execute(
                "UPDATE approvals SET status=?, decided_at=?, decided_by=?, note=? WHERE id=?",
                (status, utc_now(), actor, note, approval_id),
            )
            if approved:
                conn.execute(
                    """UPDATE tasks SET status=?, updated_at=?
                       WHERE id=? AND status=?""",
                    (TaskStatus.PENDING, utc_now(), row["task_id"], TaskStatus.WAITING_APPROVAL),
                )
            else:
                conn.execute(
                    """UPDATE tasks SET status=?, error=?, updated_at=?
                       WHERE id=? AND status=?""",
                    (TaskStatus.FAILED, f"approval rejected: {note}".rstrip(), utc_now(),
                     row["task_id"], TaskStatus.WAITING_APPROVAL),
                )
            self._append_event(conn, row["workflow_id"], row["task_id"], f"approval.{status}", {"id": approval_id})

    def list_approvals(self, workflow_id: str, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM approvals WHERE workflow_id=?"
        params: list[Any] = [workflow_id]
        if status:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY requested_at"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]
