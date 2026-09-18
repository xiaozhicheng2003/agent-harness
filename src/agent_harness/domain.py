from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


class ToolCallStatus(StrEnum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


class RiskLevel(StrEnum):
    READ_ONLY = "read_only"
    IDEMPOTENT_WRITE = "idempotent_write"
    NON_IDEMPOTENT_WRITE = "non_idempotent_write"


@dataclass(slots=True)
class Task:
    id: str
    workflow_id: str
    title: str
    role: str
    instructions: str
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    max_attempts: int = 2


@dataclass(slots=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any]
    purpose: str
    call_id: str | None = None


@dataclass(slots=True)
class AgentResult:
    output: dict[str, Any]
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ModelToolCall:
    provider_call_id: str
    name: str
    arguments: dict[str, Any]
    purpose: str = "agent requested tool call"


@dataclass(slots=True)
class ModelTurn:
    text: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)


class HarnessError(RuntimeError):
    pass


class ApprovalRequired(HarnessError):
    def __init__(self, approval_id: str, call_id: str, message: str):
        super().__init__(message)
        self.approval_id = approval_id
        self.call_id = call_id


class UnsafeRetryError(HarnessError):
    pass


class LeaseLostError(HarnessError):
    """A worker finished a side effect after losing the lease that owned it."""


class CallConflictError(HarnessError):
    pass
