from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from jsonschema import validate as validate_json_schema

from .domain import (
    ApprovalRequired, CallConflictError, LeaseLostError, RiskLevel, ToolCallStatus,
    UnsafeRetryError,
)
from .store import SQLiteStore, dumps, utc_now


ToolHandler = Callable[[dict[str, Any]], Any]


class MCPClient(Protocol):
    def call_tool(self, server: str, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    risk: RiskLevel
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "additionalProperties": True}
    )
    handler: ToolHandler | None = None
    mcp_server: str | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if (self.handler is None) == (self.mcp_server is None):
            raise ValueError("exactly one of handler or mcp_server is required")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "risk": spec.risk,
            }
            for spec in self._tools.values()
        ]


class ToolGateway:
    """Single choke point for authorization, checkpointing, routing, and audit."""

    def __init__(
        self,
        store: SQLiteStore,
        registry: ToolRegistry,
        *,
        mcp_client: MCPClient | None = None,
        worker_id: str | None = None,
        lease_seconds: int = 60,
        require_approval: set[RiskLevel] | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.mcp_client = mcp_client
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.lease_seconds = lease_seconds
        self.require_approval = require_approval or {RiskLevel.NON_IDEMPOTENT_WRITE}

    @staticmethod
    def fingerprint(arguments: dict[str, Any]) -> str:
        encoded = dumps(arguments).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def authorization_status(
        self,
        *,
        session_id: str,
        workflow_id: str,
        task_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        purpose: str,
        call_id: str,
    ) -> dict[str, str | None]:
        """Prepare an approval without executing the tool.

        This is safe to call again when a LangGraph interrupt re-enters a node.
        """
        spec = self.registry.get(tool_name)
        validate_json_schema(instance=arguments, schema=spec.input_schema)
        fingerprint = self.fingerprint(arguments)
        try:
            self._authorize_or_request(
                spec, session_id, workflow_id, task_id, call_id, fingerprint, arguments, purpose
            )
            return {"status": "approved", "approval_id": None}
        except ApprovalRequired as exc:
            return {"status": "pending", "approval_id": exc.approval_id}
        except PermissionError:
            return {"status": "rejected", "approval_id": f"approval_{call_id}"}

    def execute(
        self,
        *,
        session_id: str,
        workflow_id: str,
        task_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        purpose: str,
        call_id: str,
    ) -> Any:
        spec = self.registry.get(tool_name)
        validate_json_schema(instance=arguments, schema=spec.input_schema)
        fingerprint = self.fingerprint(arguments)

        # Authorization is deliberately evaluated before any handler/MCP side effect.
        self._authorize_or_request(
            spec, session_id, workflow_id, task_id, call_id, fingerprint, arguments, purpose
        )
        action, replay = self._claim(
            spec, session_id, workflow_id, task_id, call_id, fingerprint, arguments, purpose
        )
        if action == "replay":
            self.store.append_event(workflow_id, task_id, "tool.replayed", {"call_id": call_id})
            return replay

        try:
            if spec.handler is not None:
                result = spec.handler(arguments)
            else:
                if self.mcp_client is None:
                    raise RuntimeError(f"MCP client is not configured for {spec.name}")
                result = self.mcp_client.call_tool(spec.mcp_server or "", spec.name, arguments)
        except Exception as exc:
            self._record_failure(call_id, workflow_id, task_id, spec.risk, str(exc))
            raise
        return self._record_success(call_id, workflow_id, task_id, spec, result)

    def _authorize_or_request(
        self, spec: ToolSpec, session_id: str, workflow_id: str, task_id: str,
        call_id: str, fingerprint: str, arguments: dict[str, Any], purpose: str,
    ) -> None:
        if spec.risk not in self.require_approval:
            return
        approval_id = f"approval_{call_id}"
        with self.store.transaction(immediate=True) as conn:
            existing_call = conn.execute(
                "SELECT args_fingerprint, status FROM tool_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            if existing_call and existing_call["args_fingerprint"] != fingerprint:
                raise CallConflictError(f"call_id {call_id} was reused with different arguments")
            approval = conn.execute("SELECT * FROM approvals WHERE call_id=?", (call_id,)).fetchone()
            if approval and approval["args_fingerprint"] != fingerprint:
                raise CallConflictError(f"approval {approval_id} does not match arguments")
            if approval and approval["status"] == "approved":
                return
            if approval and approval["status"] == "rejected":
                raise PermissionError(f"tool call rejected: {call_id}")
            if approval is None:
                now = utc_now()
                conn.execute(
                    """INSERT INTO approvals
                    (id, workflow_id, task_id, call_id, tool_name, args_fingerprint,
                     purpose, status, requested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (approval_id, workflow_id, task_id, call_id, spec.name, fingerprint, purpose, now),
                )
                conn.execute(
                    """INSERT OR IGNORE INTO tool_calls
                    (call_id, session_id, workflow_id, task_id, tool_name, risk,
                     args_fingerprint, args_json, purpose, status, attempt, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (call_id, session_id, workflow_id, task_id, spec.name, spec.risk,
                     fingerprint, dumps(arguments), purpose, ToolCallStatus.WAITING_APPROVAL, now, now),
                )
                self.store._append_event(
                    conn, workflow_id, task_id, "approval.requested",
                    {"id": approval_id, "call_id": call_id, "tool": spec.name, "purpose": purpose},
                )
        raise ApprovalRequired(approval_id, call_id, f"approval required for {spec.name}")

    def _claim(
        self, spec: ToolSpec, session_id: str, workflow_id: str, task_id: str,
        call_id: str, fingerprint: str, arguments: dict[str, Any], purpose: str,
    ) -> tuple[str, Any]:
        now_epoch = time.time()
        with self.store.transaction(immediate=True) as conn:
            row = conn.execute("SELECT * FROM tool_calls WHERE call_id=?", (call_id,)).fetchone()
            if row:
                if row["session_id"] != session_id or row["args_fingerprint"] != fingerprint:
                    raise CallConflictError(f"call_id {call_id} identity mismatch")
                if row["status"] == ToolCallStatus.SUCCEEDED:
                    return "replay", json.loads(row["result_json"])
                active_lease = row["lease_until"] and row["lease_until"] > now_epoch
                if row["status"] == ToolCallStatus.RUNNING and active_lease:
                    raise RuntimeError(f"tool call {call_id} is already running")
                if row["status"] in (ToolCallStatus.RUNNING, ToolCallStatus.UNCERTAIN) and spec.risk == RiskLevel.NON_IDEMPOTENT_WRITE:
                    conn.execute(
                        "UPDATE tool_calls SET status=?, updated_at=? WHERE call_id=?",
                        (ToolCallStatus.UNCERTAIN, utc_now(), call_id),
                    )
                    raise UnsafeRetryError(
                        f"outcome of non-idempotent call {call_id} is uncertain; reconcile before retry"
                    )
                attempt = row["attempt"] + 1
                conn.execute(
                    """UPDATE tool_calls SET status=?, attempt=?, lease_owner=?, lease_until=?,
                       error=NULL, updated_at=? WHERE call_id=?""",
                    (ToolCallStatus.RUNNING, attempt, self.worker_id,
                     now_epoch + self.lease_seconds, utc_now(), call_id),
                )
            else:
                attempt = 1
                now = utc_now()
                conn.execute(
                    """INSERT INTO tool_calls
                    (call_id, session_id, workflow_id, task_id, tool_name, risk,
                     args_fingerprint, args_json, purpose, status, attempt, lease_owner,
                     lease_until, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (call_id, session_id, workflow_id, task_id, spec.name, spec.risk,
                     fingerprint, dumps(arguments), purpose, ToolCallStatus.RUNNING, attempt,
                     self.worker_id, now_epoch + self.lease_seconds, now, now),
                )
            self.store._append_event(
                conn, workflow_id, task_id, "tool.started",
                {"call_id": call_id, "tool": spec.name, "attempt": attempt, "risk": spec.risk},
            )
        return "execute", None

    def _record_success(
        self, call_id: str, workflow_id: str, task_id: str, spec: ToolSpec, result: Any,
    ) -> Any:
        with self.store.transaction(immediate=True) as conn:
            cursor = conn.execute(
                """UPDATE tool_calls SET status=?, result_json=?, lease_owner=NULL,
                   lease_until=NULL, updated_at=?
                   WHERE call_id=? AND lease_owner=?""",
                (ToolCallStatus.SUCCEEDED, dumps(result), utc_now(), call_id, self.worker_id),
            )
            if cursor.rowcount:
                self.store._append_event(
                    conn, workflow_id, task_id, "tool.succeeded", {"call_id": call_id}
                )
                return result

            # The lease expired while the handler was still running and another
            # worker took the call over, so a blind UPDATE would clobber whatever
            # that worker already persisted.
            stored = conn.execute(
                "SELECT status, result_json FROM tool_calls WHERE call_id=?", (call_id,)
            ).fetchone()
            recorded_status = stored["status"] if stored else None
            self.store._append_event(conn, workflow_id, task_id, "tool.lease_lost", {
                "call_id": call_id, "risk": spec.risk,
                "recorded_status": recorded_status, "discarded_result": result,
            })
            if (
                spec.risk != RiskLevel.NON_IDEMPOTENT_WRITE
                and recorded_status == ToolCallStatus.SUCCEEDED
                and stored["result_json"] is not None
            ):
                # Read-only and idempotent calls are safe to duplicate, so the
                # worker that won the lease owns the authoritative result.
                return json.loads(stored["result_json"])
        # Non-idempotent outcomes cannot be reconciled automatically: surface it.
        raise LeaseLostError(
            f"lease for {call_id} was taken over during execution; "
            f"recorded status is {recorded_status}"
        )

    def _record_failure(
        self, call_id: str, workflow_id: str, task_id: str, risk: RiskLevel, error: str,
    ) -> None:
        # An exception does not prove that an external side effect did not happen.
        # Idempotent operations are safe to retry; non-idempotent ones need reconciliation.
        status = (
            ToolCallStatus.UNCERTAIN
            if risk == RiskLevel.NON_IDEMPOTENT_WRITE
            else ToolCallStatus.FAILED
        )
        with self.store.transaction(immediate=True) as conn:
            cursor = conn.execute(
                """UPDATE tool_calls SET status=?, error=?, lease_owner=NULL,
                   lease_until=NULL, updated_at=?
                   WHERE call_id=? AND lease_owner=?""",
                (status, error, utc_now(), call_id, self.worker_id),
            )
            event_type = "tool.uncertain" if status == ToolCallStatus.UNCERTAIN else "tool.failed"
            if cursor.rowcount:
                self.store._append_event(
                    conn, workflow_id, task_id, event_type, {"call_id": call_id, "error": error},
                )
                return
            # Not our lease any more: the new owner's record of this call stands,
            # and our failure is preserved in the audit trail instead of overwriting it.
            self.store._append_event(conn, workflow_id, task_id, "tool.lease_lost", {
                "call_id": call_id, "risk": risk, "discarded_error": error,
            })
