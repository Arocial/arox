from dataclasses import dataclass


@dataclass
class BackgroundTask:
    completed: bool = False
    notice: str | None = None
    observed: bool = False


class BackgroundTaskBroker:
    """Coordinate background completion, prompt delivery, and explicit reads."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._notices: list[str] = []

    def register(self, key: str) -> None:
        self._tasks[key] = BackgroundTask()

    def complete(self, key: str, notice: str) -> None:
        task = self._tasks.get(key)
        if task is None:
            self._notices.append(notice)
            return
        task.notice = notice
        task.completed = True

    def observe(self, key: str) -> None:
        task = self._tasks.get(key)
        if task is not None:
            task.observed = True

    def notify(self, notice: str) -> None:
        if notice:
            self._notices.append(notice)

    def drain_notices(self) -> list[str]:
        notices, self._notices = self._notices, []
        for task in self._tasks.values():
            if task.completed and not task.observed and task.notice:
                notices.append(task.notice)
                task.observed = True
        self._discard_observed()
        return notices

    def _discard_observed(self) -> None:
        self._tasks = {
            key: task
            for key, task in self._tasks.items()
            if not (task.observed and task.completed)
        }
