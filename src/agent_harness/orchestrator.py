from __future__ import annotations

import uuid
from collections.abc import Mapping

from .agents import Agent
from .context import ContextAssembler
from .domain import (
    ApprovalRequired, LeaseLostError, TaskStatus, UnsafeRetryError, WorkflowStatus,
)
from .store import SQLiteStore


class Orchestrator:
    """Persistent DAG scheduler. Calling run() again safely resumes a workflow."""

    def __init__(
        self,
        store: SQLiteStore,
        agents: Mapping[str, Agent],
        context: ContextAssembler | None = None,
        *,
        worker_id: str | None = None,
        task_lease_seconds: float = 300.0,
    ) -> None:
        self.store = store
        self.agents = agents
        self.context = context or ContextAssembler(store)
        self.worker_id = worker_id or f"orchestrator_{uuid.uuid4().hex[:8]}"
        self.task_lease_seconds = task_lease_seconds

    def run(self, workflow_id: str) -> WorkflowStatus:
        self.store.resume_approved_tasks(workflow_id)
        # A task still marked running whose lease expired belongs to a process
        # that died; without this the workflow would never make progress again.
        self.store.reclaim_expired_tasks(workflow_id)
        self.store.set_workflow_status(workflow_id, WorkflowStatus.RUNNING)

        while True:
            ready = self.store.ready_tasks(workflow_id)
            if not ready:
                return self._settle(workflow_id)
            made_progress = False
            for task in ready:
                if not self.store.start_task(
                    task.id,
                    worker_id=self.worker_id,
                    lease_seconds=self.task_lease_seconds,
                ):
                    continue
                made_progress = True
                try:
                    agent = self.agents[task.role]
                    result = agent.run(task, self.context.assemble(workflow_id, task))
                    for artifact in result.artifacts:
                        self.store.add_artifact(workflow_id, task.id, artifact)
                    self.store.finish_task(task.id, result.output)
                except ApprovalRequired:
                    self.store.wait_task_for_approval(task.id)
                except (UnsafeRetryError, LeaseLostError) as exc:
                    # Both mean the side-effect ledger no longer trusts this
                    # worker's view of the world; retrying would guess.
                    self.store.fail_task(task.id, str(exc), retryable=False)
                except PermissionError as exc:
                    self.store.fail_task(task.id, str(exc), retryable=False)
                except Exception as exc:
                    self.store.fail_task(task.id, str(exc), retryable=True)
            if not made_progress:
                return self._settle(workflow_id)

    def _settle(self, workflow_id: str) -> WorkflowStatus:
        tasks = self.store.list_tasks(workflow_id)
        statuses = {task.status for task in tasks}
        if statuses and statuses <= {TaskStatus.SUCCEEDED}:
            status = WorkflowStatus.SUCCEEDED
        elif TaskStatus.WAITING_APPROVAL in statuses:
            status = WorkflowStatus.WAITING_APPROVAL
        elif TaskStatus.FAILED in statuses:
            status = WorkflowStatus.FAILED
        elif TaskStatus.RUNNING in statuses:
            # Another worker holds a live lease on a task; it is still in progress.
            status = WorkflowStatus.RUNNING
        elif any(task.status == TaskStatus.PENDING for task in tasks):
            # No ready nodes with pending work means a failed dependency or invalid DAG.
            status = WorkflowStatus.FAILED
        else:
            status = WorkflowStatus.PENDING
        self.store.set_workflow_status(workflow_id, status)
        return status
