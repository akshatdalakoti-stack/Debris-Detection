"""The shared Redis handle, and a breaker so an outage stays cheap.

Three things need it - the job queue, the enrichment cache and the sign-in rate
limiter - and none of them should have to import each other to get a
connection. redis-py is lazy, so constructing this does not connect; the first
command does, and every caller here is written to carry on without it.

Carrying on without it is not free, though. Each call to a Redis that is not
answering costs a connection timeout, and these are the paths that run on every
sign-in and every detection. Timeouts bound one attempt; the breaker stops the
attempts happening at all for a while once it is clear nobody is home.
"""

from __future__ import annotations

import logging
import time

import redis

from ..config import settings

log = logging.getLogger(__name__)

# Far longer than a healthy Redis needs, short enough that an unhealthy one
# degrades rather than blocking. Without these redis-py waits on the OS default.
CONNECT_TIMEOUT_SECONDS = 0.5
COMMAND_TIMEOUT_SECONDS = 0.5

# How long to stop trying after a failure, before probing again.
BREAKER_COOLDOWN_SECONDS = 30.0

redis_client = redis.Redis.from_url(
    settings.redis_url,
    socket_connect_timeout=CONNECT_TIMEOUT_SECONDS,
    socket_timeout=COMMAND_TIMEOUT_SECONDS,
)

_closed_until = 0.0


def cache_is_up() -> bool:
    """False while the breaker is open - skip Redis, go straight to fallback.

    A plain try/except around each call still pays the timeout every time, so a
    Redis outage turned a half-second into a half-second per request. This
    makes the first failure the expensive one and the rest free until the
    cooldown lapses and the next call probes again.
    """
    return time.monotonic() >= _closed_until


def note_cache_failure(exc: Exception | None = None) -> None:
    global _closed_until
    first = cache_is_up()
    _closed_until = time.monotonic() + BREAKER_COOLDOWN_SECONDS
    if first:
        # Once per outage, not once per request.
        log.warning("redis unreachable (%s) - falling back for %gs",
                    exc, BREAKER_COOLDOWN_SECONDS)


def note_cache_success() -> None:
    global _closed_until
    _closed_until = 0.0
