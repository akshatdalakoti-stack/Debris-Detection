"""Sensor check: is this imagery the kind the loaded models were trained on.

A model is accurate on its own sensor and close to useless on the other - a
measured 1-in-39 hit rate when the small-debris (forward-looking) model is
pointed at side-scan. That cuts both ways, and it is why this check exists.

What actually ships is two side-scan heads - SCTD wrecks and Ghost Pot ghost
gear - so there is no forward-looking head to route to, and the choice the
original version of this docstring described (route, do not ensemble) is not
the choice in front of the pipeline. Running two side-scan heads over
side-scan imagery is a different decision, and a defensible one: they find
different objects.

The live risk is the other one. A forward-looking frame handed to this system
gets two side-scan heads and used to get them silently. run_inference now asks
this module first and reports the answer, so a mismatch is visible rather than
arriving as a confident, wrong detection list.

Routing uses image geometry. Forward-looking sonar renders as a fan with the
transducer at the apex, leaving large black corners. Side-scan fills the frame
edge to edge. Measuring how much of the corner area is near-black separates
them cleanly.

Measured on 226 images: forward-looking 100% correct, side-scan 92%, 98.7%
overall at the default threshold. The failures are side-scan frames that
arrive letterboxed with dark borders, so `detect_sensor` reports its margin
and every caller can override the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIDESCAN = "sidescan"
FLS = "fls"

CORNER_FRACTION = 0.18   # how much of each corner to sample
DARK_LEVEL = 20          # 0-255; below this counts as "no return"
FAN_THRESHOLD = 0.70     # corner darkness above which it reads as a fan


@dataclass
class Routing:
    sensor: str
    corner_darkness: float
    confident: bool
    reason: str


def detect_sensor(image: np.ndarray | str | Path,
                  threshold: float = FAN_THRESHOLD) -> Routing:
    """Decide which sensor produced this image.

    Returns the choice plus the evidence, so a caller can surface a low-margin
    decision to the operator instead of silently routing to the wrong model.
    """
    if not isinstance(image, np.ndarray):
        from PIL import Image
        image = np.array(Image.open(image).convert("L"))

    h, w = image.shape[:2]
    ch, cw = max(1, int(h * CORNER_FRACTION)), max(1, int(w * CORNER_FRACTION))
    corners = np.concatenate([
        image[:ch, :cw].ravel(), image[:ch, -cw:].ravel(),
        image[-ch:, :cw].ravel(), image[-ch:, -cw:].ravel(),
    ])
    darkness = float((corners < DARK_LEVEL).mean())

    sensor = FLS if darkness >= threshold else SIDESCAN
    margin = abs(darkness - threshold)
    confident = margin >= 0.05

    if sensor == FLS:
        reason = (f"{darkness:.0%} of corner area is empty - consistent with a "
                  f"forward-looking fan")
    else:
        reason = (f"only {darkness:.0%} of corner area is empty - image fills "
                  f"the frame, consistent with a side-scan waterfall")
    if not confident:
        reason += " (low margin - override with --sensor if this is wrong)"

    return Routing(sensor=sensor, corner_darkness=round(darkness, 3),
                   confident=confident, reason=reason)
