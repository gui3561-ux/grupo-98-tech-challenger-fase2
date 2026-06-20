import threading
import uuid
from datetime import datetime, timezone
from enum import StrEnum


class Status(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    def __init__(self, job_id: str, step: str):
        self.job_id = job_id
        self.step = step
        self.status = Status.PENDING
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error: str | None = None


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, step: str) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = Job(job_id, step)
        return job_id

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def has_running_job(self, step: str) -> bool:
        with self._lock:
            return any(
                j.step == step and j.status == Status.RUNNING
                for j in self._jobs.values()
            )

    def run_in_background(self, job_id: str, target: callable) -> None:
        def wrapper():
            job = self._jobs[job_id]
            with self._lock:
                job.status = Status.RUNNING
                job.started_at = datetime.now(timezone.utc)
            try:
                target()
                with self._lock:
                    job.status = Status.COMPLETED
                    job.completed_at = datetime.now(timezone.utc)
            except Exception as e:
                with self._lock:
                    job.status = Status.FAILED
                    job.completed_at = datetime.now(timezone.utc)
                    job.error = str(e)

        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
