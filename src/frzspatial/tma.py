"""Tissue microarrays: construction keys, FOV-to-core assignment and case metadata.

A TMA key is a text matrix oriented like the slide's core grid: entry ``[r - 1, c - 1]`` is the sample in core
``C<c>R<r>`` (column ``c`` from the left, row ``r`` from the top of the slide). :func:`assign_cores` gives every FOV
(and so every cell) a core and its sample, with the algorithm of gbspatial's ``assign_fovs_to_cores()`` (Giuseppe
Barisano); :func:`plot_core_grid` draws the row and column sweeps used to check the key's orientation.
"""

import re
import warnings

import numpy as np
import pandas as pd

from ._utils import _log, _set_rows, _slide_rows

__all__ = ["key_matrix", "assign_cores", "plot_core_grid", "cases", "add_clinical"]


def _as_text(v):
    """A spreadsheet cell as text, as R reads it: 37128.0 -> "37128", empty -> ""."""
    if pd.isna(v):
        return ""
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def key_matrix(key):
    """A TMA key as a text matrix, empty positions named ``control1``, ``control2``, ... (numbered down the columns).

    Parameters
    ----------
    key
        DataFrame or array, rows = core rows and columns = core columns of the slide, e.g. read from a spreadsheet
        and reoriented (``.iloc[::-1]``, ``.T``) to match the slide.

    Returns
    -------
    ndarray of str
        The key; ``key[r - 1, c - 1]`` is the sample of core ``C<c>R<r>``.
    """
    m = pd.DataFrame(key).map(_as_text).to_numpy(object)
    empty = np.flatnonzero(m.T == "")
    m.T.flat[empty] = [f"control{i + 1}" for i in range(len(empty))]
    return m.astype(str)  # after filling: a str array is as wide as its longest entry


def _check_orientation(key, x, y, fov_size, slide):
    """Warn when the key's shape does not fit the FOV grid on the slide (it may need transposing)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return
    x, y = x[ok], y[ok]
    n_fov_cols = len(np.unique(np.round((x - x.min()) / fov_size)))
    n_fov_rows = len(np.unique(np.round((y - y.min()) / fov_size)))
    n_rows, n_cols = key.shape
    tag = f"[{slide}] " if slide else ""
    if n_cols > n_fov_cols or n_rows > n_fov_rows:
        hint = "the transposed key fits, try key.T" if n_rows <= n_fov_cols and n_cols <= n_fov_rows else \
            "check how the key was read"
        warnings.warn(f"{tag}TMA key is {n_rows} rows x {n_cols} columns, but the slide spans only ~{n_fov_cols} FOV "
                      f"columns x {n_fov_rows} FOV rows: {hint}", stacklevel=3)
        return
    asp = n_fov_cols / n_fov_rows
    if n_rows != n_cols and abs(np.log((n_rows / n_cols) / asp)) + np.log(1.3) < abs(np.log((n_cols / n_rows) / asp)):
        warnings.warn(f"{tag}TMA key orientation ({n_rows} x {n_cols}) fits the slide (~{n_fov_cols} x {n_fov_rows} "
                      "FOVs) poorly; the transposed key fits much better. Check with plot_core_grid().", stacklevel=3)


def _nearest(px, py, ax, ay):
    """Index of the nearest anchor for each point (squared distance, first anchor on ties, as R's max.col)."""
    d = (px[:, None] - ax[None, :]) ** 2 + (py[:, None] - ay[None, :]) ** 2
    return d.argmin(axis=1)


def assign_cores(sdata, key, slide="", core_drift_tolerance=0.4, min_cells=30, fov_size=4256, fov_overrides=None,
                 table="table"):
    """Assign each FOV, and so each cell, to a TMA core and its sample.

    The algorithm of gbspatial's ``assign_fovs_to_cores()``:

    1. A regular grid of anchors, one per key position, spans the FOVs.
    2. Each cell goes to its nearest anchor; each anchor moves to the mean of its cells, and is dropped with fewer
       than `min_cells` cells or a drift beyond `core_drift_tolerance` of a grid step.
    3. Each FOV takes the core most of its cells are nearest to; FOVs straddling a grid line take the majority core
       of the FOVs within 1.6 FOV widths, the others the core of their grid cell.
    4. FOVs too far from their core's anchor (or whose core was dropped) get no core.

    Parameters
    ----------
    sdata
        A single slide, or a merged object (:func:`merge_slides`) whose obs ``slide`` holds `slide`: then only that
        slide's cells are assigned and the other slides' rows are left as they are. obs needs ``fov``,
        ``CenterX/Y_global_px`` and ``CenterX/Y_local_px``.
    key
        TMA key oriented like the slide (:func:`key_matrix`).
    slide
        Slide name, for messages and plots; with a merged object, the slide to assign.
    core_drift_tolerance, min_cells, fov_size
        As in gbspatial: anchor drift limit (fraction of a grid step), minimum cells per core, FOV width in px.
    fov_overrides
        Manual fixes applied afterwards, ``{fov: core_str}`` (e.g. ``{109: "C4R10"}``); the core's position and
        sample come from the key.
    table
        Table element.

    Returns
    -------
    None
        Works in place: sets obs ``core_str``, ``core_col``, ``core_row`` and ``sample_id`` (NA for FOVs without a
        core), and keeps the per-FOV table in ``uns["tma_fovs"][slide]`` (saved with the object): ``{"fovs":
        DataFrame, "grid": dict}``, one row per FOV (``fov``, ``core_str``, ``core_col``, ``core_row``,
        ``sample_id``, its box and whether it straddles the grid) and the grid :func:`plot_core_grid` draws.
    """
    full = sdata.tables[table].obs
    rows = _slide_rows(full, slide)  # None: a single slide
    obs = full if rows is None else full[rows]
    key = np.asarray(key, dtype=object)
    n_rows, n_cols = key.shape
    fov = obs["fov"].to_numpy().astype(int)
    cx = obs["CenterX_global_px"].to_numpy(float)
    cy = obs["CenterY_global_px"].to_numpy(float)

    # 0. FOV positions from the cells (AtoMx: x = global - local, y = global + local) and the core grid
    fovs = pd.DataFrame({"fov": fov, "x": cx - obs["CenterX_local_px"].to_numpy(float),
                         "y": cy + obs["CenterY_local_px"].to_numpy(float)}).groupby("fov").median().reset_index()
    _check_orientation(key, cx, cy, fov_size, slide)
    xmin, xmax = fovs["x"].min(), fovs["x"].max() + fov_size
    ymin, ymax = fovs["y"].min() - fov_size, fovs["y"].max()
    xstep, ystep = (xmax - xmin) / n_cols, (ymax - ymin) / n_rows
    v_lines, h_lines = xmin + np.arange(n_cols + 1) * xstep, ymin + np.arange(n_rows + 1) * ystep
    a_col = np.tile(np.arange(1, n_cols + 1), n_rows)
    a_row = np.repeat(np.arange(1, n_rows + 1), n_cols)
    a_x, a_y = xmin + (a_col - 0.5) * xstep, ymax - (a_row - 0.5) * ystep
    a_str = np.array([f"C{c}R{r}" for c, r in zip(a_col, a_row)])

    # 1. Anchors refined to the mean of their cells; too few cells or too much drift drops them
    first = _nearest(cx, cy, a_x, a_y)
    ref = pd.DataFrame({"a": first, "x": cx, "y": cy}).groupby("a").agg(x=("x", "mean"), y=("y", "mean"),
                                                                       n=("x", "size"))
    ref = ref.reindex(np.arange(len(a_str)))
    valid = (ref["n"].to_numpy() >= min_cells) & (np.abs(ref["x"].to_numpy() - a_x) <= xstep * core_drift_tolerance) \
        & (np.abs(ref["y"].to_numpy() - a_y) <= ystep * core_drift_tolerance)
    if not valid.any():
        raise ValueError(f"no valid cores found for slide {slide}")
    f_idx = np.flatnonzero(valid)
    f_x, f_y, f_str = ref["x"].to_numpy()[f_idx], ref["y"].to_numpy()[f_idx], a_str[f_idx]

    # 2. Each FOV's vote: the valid core most of its cells are nearest to (ties: first core name)
    vote = pd.DataFrame({"fov": fov, "core": f_str[_nearest(cx, cy, f_x, f_y)]}).value_counts().rename("n") \
        .reset_index().sort_values(["fov", "n", "core"], ascending=[True, False, True]).drop_duplicates("fov")
    fovs["vote"] = fovs["fov"].map(vote.set_index("fov")["core"])

    # 3. FOV boxes; straddling FOVs take the majority vote of their neighbourhood, the others their grid cell
    fovs["xmin"], fovs["xmax"] = fovs["x"], fovs["x"] + fov_size
    fovs["ymin"], fovs["ymax"] = fovs["y"] - fov_size, fovs["y"]
    xc, yc = (fovs["x"] + fov_size / 2).to_numpy(), (fovs["y"] - fov_size / 2).to_numpy()
    crosses = lambda lines, lo, hi: ((lines[None, :] > lo[:, None] + 0.001) & (lines[None, :] < hi[:, None] - 0.001)).any(1)
    fovs["straddles_grid"] = crosses(v_lines, fovs["xmin"].to_numpy(), fovs["xmax"].to_numpy()) | \
        crosses(h_lines, fovs["ymin"].to_numpy(), fovs["ymax"].to_numpy())
    geometric = a_str[_nearest(xc, yc, a_x, a_y)]
    prov = fovs["vote"].fillna(pd.Series(f_str[_nearest(xc, yc, f_x, f_y)])).to_numpy()
    centre = pd.DataFrame({"fov": fov, "x": cx, "y": cy}).groupby("fov").mean().reindex(fovs["fov"]).to_numpy()
    near = np.hypot(*(centre[:, None, :] - centre[None, :, :]).transpose(2, 0, 1)) <= fov_size * 1.6
    assigned = []
    for k in range(len(fovs)):
        if fovs["straddles_grid"].iat[k]:
            names, counts = np.unique(prov[near[k]].astype(str), return_counts=True)  # sorted, as R's table()
            assigned.append(names[counts.argmax()])
        else:
            assigned.append(geometric[k])
    assigned = np.array(assigned)

    # 4. No core when the FOV is far from its core's anchor, or the core was dropped
    pos = {s: i for i, s in enumerate(f_str)}
    j = np.array([pos.get(s, -1) for s in assigned])
    dist = np.where(j >= 0, np.hypot(xc - f_x[j], yc - f_y[j]), np.nan)
    keep = dist <= np.hypot(xstep, ystep) * 0.9
    fovs["core_str"] = np.where(keep, assigned, None)
    fovs["core_col"] = np.where(keep, a_col[f_idx][j], np.nan)
    fovs["core_row"] = np.where(keep, a_row[f_idx][j], np.nan)

    # 5. Manual fixes, then the sample of each core from the key
    for f, core in (fov_overrides or {}).items():
        m = re.fullmatch(r"C(\d+)R(\d+)", core)
        if m is None or not (fovs["fov"] == f).any():
            raise ValueError(f"fov_overrides: FOV {f} -> {core!r} (FOV not on the slide, or core not C<col>R<row>)")
        fovs.loc[fovs["fov"] == f, ["core_str", "core_col", "core_row"]] = [core, int(m[1]), int(m[2])]
    has = fovs["core_str"].notna().to_numpy()
    fovs["sample_id"] = None
    fovs.loc[has, "sample_id"] = key[fovs.loc[has, "core_row"].astype(int) - 1, fovs.loc[has, "core_col"].astype(int) - 1]

    # 6. Cells take their FOV's core
    per_cell = fovs.set_index("fov").reindex(fov)
    for col, kind in (("core_str", "category"), ("core_col", "float"), ("core_row", "float"), ("sample_id", "category")):
        _set_rows(full, col, per_cell[col].to_numpy(), rows, kind)
    no_core = per_cell["core_str"].isna().to_numpy()
    _log(f"{slide}: {per_cell['core_str'].nunique()} cores; cells without a core: {int(no_core.sum())}"
         + (f" (FOVs {', '.join(map(str, sorted(set(fov[no_core]))))})" if no_core.any() else "")
         + (f"; fixed by hand: FOVs {', '.join(map(str, fov_overrides))}" if fov_overrides else ""))

    fovs = fovs.drop(columns=["x", "y", "vote"])
    fovs["core_str"] = pd.Categorical(fovs["core_str"])  # categorical: NA-safe when written to Zarr
    fovs["sample_id"] = pd.Categorical(fovs["sample_id"].where(fovs["sample_id"].notna(), None).astype(object))
    fovs.attrs = {"slide": slide, "v_lines": v_lines, "h_lines": h_lines, "n_rows": n_rows, "n_cols": n_cols,
                  "x_breaks": xmin + (np.arange(1, n_cols + 1) - 0.5) * xstep,
                  "y_breaks": ymax - (np.arange(1, n_rows + 1) - 0.5) * ystep}
    grid = {k: v for k, v in fovs.attrs.items() if k != "slide"}
    fovs.attrs = {}  # the grid is stored next to the table
    sdata.tables[table].uns.setdefault("tma_fovs", {})[slide or "slide"] = {"fovs": fovs, "grid": grid}


def _stored_fovs(sdata, slide, table):
    """The per-FOV table assign_cores kept in uns for `slide`, grid in ``.attrs``; None if there is none."""
    stored = getattr(sdata.tables[table], "uns", {}).get("tma_fovs", {})
    if slide is None and len(stored) == 1:
        slide = next(iter(stored))
    if slide not in stored:
        return None
    fovs = stored[slide]["fovs"].copy()
    grid = {k: (v.item() if np.ndim(v) == 0 and hasattr(v, "item") else v) for k, v in stored[slide]["grid"].items()}
    fovs.attrs = {"slide": slide, **grid}  # Zarr gives scalars back as 0-d arrays
    return fovs


def plot_core_grid(fovs=None, by="row", sdata=None, ax=None, table="table", fontsize=4, size=1, slide=None):
    """The core grid with each FOV coloured by its core row, column or core.

    With ``by="row"`` or ``"column"`` each FOV is a box labelled ``<fov> [<core>]``: sweeping down the rows (or
    across the columns), the colours should step monotonically. With ``by="core"`` the cells are drawn, one hue per
    core and one shade of it per FOV, with core names in bold and FOV numbers in grey. Either way, a FOV out of step
    belongs to another core (fix it with ``fov_overrides`` in :func:`assign_cores`).

    Parameters
    ----------
    fovs
        A per-FOV table with its grid in ``.attrs`` (default: the one :func:`assign_cores` kept in `sdata`'s
        ``uns["tma_fovs"]`` for `slide`). It gives the FOV boxes of ``"row"`` and ``"column"`` and the grid lines and
        C/R labels; with ``"core"`` the cores come from obs ``core_str``, and without a table the cells are drawn with
        no grid.
    by
        ``"row"``, ``"column"`` or ``"core"``.
    sdata
        The slide, to draw its cells: a sample in grey behind the FOV boxes (optional), or all of them with
        ``by="core"`` (required, after :func:`assign_cores`). A merged object works too: only the cells with
        ``obs["slide"]`` equal to `slide` are drawn.
    ax
        Matplotlib axes (default: a new figure, 7 x 10 in for rows and columns, matplotlib's size for cores).
    fontsize
        Size of the FOV labels; core names are 1.6 times and the title 2.5 times larger.
    size
        Marker size of the cells with ``by="core"``.
    slide
        The slide to draw from a merged `sdata` (default: the slide of `fovs`).

    Returns
    -------
    Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if by not in ("row", "column", "core"):
        raise ValueError(f"by must be 'row', 'column' or 'core', not {by!r}")
    if by == "core" and sdata is None:
        raise ValueError("by='core' draws the cells: give sdata")
    if fovs is None and sdata is not None:
        fovs = _stored_fovs(sdata, slide, table)
    if by != "core" and fovs is None:
        raise ValueError(f"by={by!r} draws the FOV boxes: give the sdata assign_cores worked on (or fovs)")
    g = fovs.attrs if fovs is not None else {}
    slide = slide or g.get("slide")
    if ax is None:
        _, ax = plt.subplots(figsize=None if by == "core" else (7, 10))
    obs = None
    if sdata is not None:
        obs = sdata.tables[table].obs
        if "slide" in obs and slide:
            obs = obs[(obs["slide"].astype(str) == slide).to_numpy()]
    if by == "core":
        _cells_by_core(ax, obs, fontsize, size)
        ax.set_title(f"{slide}: FOVs by core" if slide else "FOVs by core", fontsize=2.5 * fontsize)
        if fovs is None:  # no grid: just the cells, row 1 at the top as in the key and the grid plots
            ax.set_aspect("equal")
            ax.set_axis_off()
            return ax
    elif obs is not None:
        cells = obs.groupby("fov", observed=True).sample(n=200, replace=True, random_state=0) \
            .drop_duplicates()[["CenterX_global_px", "CenterY_global_px"]].to_numpy()
        ax.scatter(*cells.T, s=0.2, c="0.6", alpha=0.4, lw=0)
    for x in g["v_lines"]:
        ax.axvline(x, c="0.4", ls="--", alpha=0.6, lw=0.8)
    for y in g["h_lines"]:
        ax.axhline(y, c="0.4", ls="--", alpha=0.6, lw=0.8)
    colours = plt.get_cmap("tab20").colors
    col = "core_row" if by == "row" else "core_col"
    for _, f in fovs.iterrows() if by != "core" else ():
        c = colours[int(f[col] - 1) % 20] if pd.notna(f[col]) else "none"
        ax.add_patch(Rectangle((f["xmin"], f["ymin"]), f["xmax"] - f["xmin"], f["ymax"] - f["ymin"], fc=c, ec="k",
                               alpha=0.3))
        ax.text((f["xmin"] + f["xmax"]) / 2, (f["ymin"] + f["ymax"]) / 2,
                f"{f['fov']}\n[{f['core_str'] if pd.notna(f['core_str']) else 'NA'}]", ha="center", va="center",
                fontsize=fontsize, fontweight="bold")
    ax.set_xticks(g["x_breaks"], [f"C{i}" for i in range(1, g["n_cols"] + 1)], fontweight="bold")
    ax.set_yticks(g["y_breaks"], [f"R{i}" for i in range(1, g["n_rows"] + 1)], fontweight="bold")
    ax.set_xlim(g["v_lines"][0] - 0.05 * np.ptp(g["v_lines"]), g["v_lines"][-1] + 0.05 * np.ptp(g["v_lines"]))
    ax.set_ylim(g["h_lines"][0] - 0.05 * np.ptp(g["h_lines"]), g["h_lines"][-1] + 0.05 * np.ptp(g["h_lines"]))
    ax.set_aspect("equal")
    if by != "core":
        ax.set_title(f"{slide}: FOVs by core {by}", fontsize=2.5 * fontsize)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def _cells_by_core(ax, obs, fontsize, size):
    """Every cell coloured by its core (obs ``core_str``: one hue per core, a shade per FOV); core and FOV labels."""
    import colorsys

    if "core_str" not in obs:
        raise KeyError("obs has no core_str: run assign_cores first")
    cells = pd.DataFrame({"fov": obs["fov"].to_numpy().astype(int), "x": obs["CenterX_global_px"].to_numpy(float),
                          "y": obs["CenterY_global_px"].to_numpy(float),
                          "core": obs["core_str"].astype(object).to_numpy()})
    fov_core = cells.dropna(subset=["core"]).drop_duplicates("fov").sort_values("fov")
    shade = {}
    for i, (_, f) in enumerate(fov_core.groupby("core")):
        for v, fov in zip(np.linspace(0.45, 1, len(f)), f["fov"]):
            shade[fov] = colorsys.hsv_to_rgb((i * 0.618) % 1, 0.75, v)  # golden-ratio hues: neighbours differ
    ax.scatter(cells["x"], cells["y"], s=size, c=[shade.get(f, (0.85, 0.85, 0.85)) for f in cells["fov"]], lw=0,
               rasterized=True)
    for fov, (x, y) in cells.groupby("fov")[["x", "y"]].median().iterrows():
        ax.text(x, y, fov, ha="center", va="top", fontsize=fontsize, color="dimgrey")
    for core, (x, y) in cells.dropna(subset=["core"]).groupby("core")[["x", "y"]].median().iterrows():
        ax.text(x, y, core, ha="center", va="bottom", fontsize=1.6 * fontsize, fontweight="bold")


def cases(slide, sample_id, group_mn, sid=None):
    """Case table rows: one per case, for :func:`add_clinical`.

    Parameters
    ----------
    slide, sample_id, group_mn
        One value or one per case. ``group_mn`` is the detailed group (e.g. ``"PLA2R"``, ``"Normal"``).
    sid
        Short case name (default: `sample_id`).

    Returns
    -------
    DataFrame
        ``slide``, ``sample_id`` (as text), ``group_mn``, ``group`` (``Normal``, ``Placenta``, otherwise ``MN``)
        and ``sid``.
    """
    col = lambda v: v if np.isscalar(v) else np.asarray(v, dtype=object)
    d = pd.DataFrame({"slide": col(slide), "sample_id": pd.Series(col(sample_id)).map(_as_text),
                      "group_mn": col(group_mn)})
    d["group"] = d["group_mn"].where(d["group_mn"].isin(["Normal", "Placenta"]), "MN")
    d["sid"] = d["sample_id"] if sid is None else col(sid)
    return d


def add_clinical(sdata, slide, clinical, slides=None, table="table"):
    """Add the slide name and each core's case metadata to obs, in place.

    Sets ``slide`` (categories: `slides`), ``core_id`` (``<slide>_<core_str>``) and ``group``, ``group_mn`` and
    ``sid`` matched on ``sample_id`` (NA for cores without a known case), and prints the groups and the unknown
    cases.

    Parameters
    ----------
    sdata
        A single slide after :func:`assign_cores`, or a merged object (:func:`merge_slides`) whose obs ``slide``
        holds `slide`: then only that slide's rows are set, and the other slides' rows are left as they are.
    slide
        Its name.
    clinical
        Case table (:func:`cases`), all slides; rows of other slides are ignored.
    slides
        All slide names, as categories of ``slide`` (so that merged slides share them). Default: just `slide`, or
        the merged object's slides.
    """
    obs = sdata.tables[table].obs
    rows = _slide_rows(obs, slide)  # None: a single slide
    here = obs if rows is None else obs[rows]
    hit = here[["sample_id"]].astype(str).merge(clinical[clinical["slide"] == slide], how="left", on="sample_id")
    if rows is None:
        obs["slide"] = pd.Categorical([slide] * len(obs), categories=list(slides or [slide]))
    elif slides is not None:
        obs["slide"] = pd.Categorical(obs["slide"].astype(str), categories=list(slides))
    _set_rows(obs, "core_id", (slide + "_" + here["core_str"].astype(str)).to_numpy(), rows, "category")
    for c in ["group", "group_mn", "sid"]:
        _set_rows(obs, c, hit[c].to_numpy(), rows, "category")
    unknown = sorted(set(here.loc[hit["group"].isna().to_numpy(), "sample_id"].dropna().astype(str)))
    groups = pd.Series(hit["group_mn"].to_numpy()).value_counts(dropna=False).to_dict()
    print(f"{slide}: {groups}; cases without clinical data:", ", ".join(unknown) or "none")
