from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class LatestItem(Generic[T]):
    sequence: int
    value: T


class LatestItemMailbox(Generic[T]):
    """Thread-safe one-slot mailbox that overwrites stale pending work."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: LatestItem[T] | None = None
        self._closed = False
        self._received = 0
        self._overwritten = 0

    @property
    def received(self) -> int:
        with self._condition:
            return self._received

    @property
    def overwritten(self) -> int:
        with self._condition:
            return self._overwritten

    def put(self, value: T) -> int:
        with self._condition:
            if self._closed:
                raise RuntimeError("mailbox is closed")
            self._received += 1
            if self._pending is not None:
                self._overwritten += 1
            item = LatestItem(sequence=self._received, value=value)
            self._pending = item
            self._condition.notify()
            return item.sequence

    def take(self) -> LatestItem[T] | None:
        with self._condition:
            while self._pending is None and not self._closed:
                self._condition.wait()
            if self._pending is None:
                return None
            item = self._pending
            self._pending = None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
