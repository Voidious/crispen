from __future__ import annotations
import threading
from .exceptions import _ApiTimeout


def _run_with_timeout(func, timeout, *args, **kwargs):
    """Run *func* in a daemon thread; raise _ApiTimeout if it doesn't finish.

    This enforces a hard wall-clock limit that is not affected by OS-level
    blocking (e.g. DNS resolution) which application-layer timeouts cannot
    interrupt.
    """
    result: list = [None]
    exc: list = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise _ApiTimeout(f"API call exceeded {timeout}s hard limit")
    if exc[0] is not None:
        raise exc[0]
    return result[0]
