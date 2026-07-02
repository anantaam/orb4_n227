"""
Retry critical calls with exponential backoff (3 attempts by default).
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    logger: logging.Logger | None = None,
    operation_name: str = "operation",
) -> T:
    """
    Call fn() up to max_attempts times. On exception, sleep base_delay * 2^attempt and retry.
    Re-raises the last exception if all attempts fail.
    """
    log = logger or logging.getLogger(__name__)
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= max_attempts - 1:
                log.error("%s failed after %s attempts: %s", operation_name, max_attempts, e)
                raise
            delay = base_delay_seconds * (2**attempt)
            log.warning(
                "%s attempt %s/%s failed: %s; retrying in %.1fs",
                operation_name,
                attempt + 1,
                max_attempts,
                e,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
