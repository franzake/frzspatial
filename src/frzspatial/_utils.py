"""Shared constants and the progress log."""

import time

import numpy as np
import pandas as pd

#: CosMx pixel size, in µm per AtoMx global pixel.
PX_UM = 0.120280945

#: Set to False to silence frzspatial's progress messages.
VERBOSE = True

_T0 = time.time()


def _reset_clock():
    global _T0
    _T0 = time.time()


def _slide_rows(obs, slide):
    """Rows of `slide` when obs holds several slides (a merged object), else None (all rows are the slide)."""
    if not slide or "slide" not in obs or (obs["slide"].astype(str) == slide).all():
        return None
    rows = (obs["slide"].astype(str) == slide).to_numpy()
    if not rows.any():
        raise ValueError(f"no cells with obs['slide'] == {slide!r}")
    return rows


def _set_rows(obs, col, values, rows, kind="float", categories=None):
    """Set obs `col` to `values` on `rows` (None: all rows), keeping the other rows' values.

    `kind` is ``"float"``, ``"bool"`` (rows never set are False) or ``"category"`` (in the order of `categories`
    when given).
    """
    if rows is not None:
        fill = False if kind == "bool" else None
        old = obs[col].astype(object).to_numpy(copy=True) if col in obs else np.full(len(obs), fill, dtype=object)
        old[rows] = values
        values = old
    if kind == "category":
        v = pd.Series(values, dtype=object)
        obs[col] = pd.Categorical(v.where(v.notna(), None), categories=categories)
    elif kind == "bool":
        obs[col] = pd.Series(values, dtype=object).fillna(False).to_numpy(bool)
    else:
        obs[col] = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)


def _log(msg):
    if VERBOSE:
        print(f"[frzspatial {time.time() - _T0:5.0f} s] {msg}", flush=True)
