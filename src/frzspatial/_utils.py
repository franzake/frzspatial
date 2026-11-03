"""Shared constants and the progress log."""

import time

#: CosMx pixel size, in µm per AtoMx global pixel.
PX_UM = 0.120280945

#: Set to False to silence frzspatial's progress messages.
VERBOSE = True

_T0 = time.time()


def _reset_clock():
    global _T0
    _T0 = time.time()


def _log(msg):
    if VERBOSE:
        print(f"[frzspatial {time.time() - _T0:5.0f} s] {msg}", flush=True)
