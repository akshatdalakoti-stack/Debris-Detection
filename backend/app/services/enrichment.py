"""Depth, biodiversity and port distance for a position, with a cache in front.

enrich_detection() reaches out to GEBCO and OBIS. That is fine once; it is not
fine once per detection in a survey, and the recovery planner did it once per
hazard in a day plan with nothing in between - a 40-hazard plan meant 80 network
round trips before the endpoint answered.

Two layers:

    process  a small dict, for the same coordinate repeating inside one request
    redis    shared between the API and the workers, seven days

The values behind it are static - a bathymetry grid and an occurrence database -
so a stale entry is not a concern the way a stale price would be.
"""

from __future__ import annotations

import json
import logging
from threading import Lock

import redis

from ..config import settings
from ml.enrich import Context, enrich_detection, to_dict as context_to_dict

log = logging.getLogger(__name__)

redis_client = redis.Redis.from_url(settings.redis_url)

CACHE_TTL_SECONDS = 86_400 * 7
# ~4 decimal places is about 11 m at the equator, finer than the registry's own
# 25 m matching tolerance, so two detections of one object share a key.
KEY_PRECISION = 4
_PROCESS_CACHE_MAX = 4096

_process_cache: dict[tuple[float, float], Context | None] = {}
_process_lock = Lock()

# Redis being down should not turn every lookup into a network call to GEBCO,
# and it should not fill the log with one line per detection either. Logged
# once per process, then the code carries on without the shared layer.
_redis_warned = False


def _warn_once(exc: Exception) -> None:
    global _redis_warned
    if not _redis_warned:
        _redis_warned = True
        log.warning("enrichment cache unavailable (%s) - falling back to live "
                    "lookups for the rest of this process", exc)


def cached_enrich(lat: float | None, lon: float | None) -> Context | None:
    """Context for a position, or None when there is no position to enrich."""
    if lat is None or lon is None:
        return None

    key_parts = (round(lat, KEY_PRECISION), round(lon, KEY_PRECISION))
    with _process_lock:
        if key_parts in _process_cache:
            return _process_cache[key_parts]

    key = f"enrich:{key_parts[0]}:{key_parts[1]}"
    context: Context | None = None
    try:
        cached = redis_client.get(key)
        if cached:
            context = Context(**json.loads(cached))
    except Exception as exc:                     # redis down, or a stale shape
        _warn_once(exc)

    if context is None:
        context = enrich_detection(lat, lon, navigation_is_real=True)
        if context is not None:
            try:
                redis_client.setex(key, CACHE_TTL_SECONDS,
                                   json.dumps(context_to_dict(context)))
            except Exception as exc:
                _warn_once(exc)

    with _process_lock:
        if len(_process_cache) >= _PROCESS_CACHE_MAX:
            # Not an LRU. This only has to stop one very long survey from
            # growing the dict without bound; which entries survive a reset
            # does not matter when every miss is just a cache lookup away.
            _process_cache.clear()
        _process_cache[key_parts] = context
    return context
