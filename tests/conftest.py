"""A tiny synthetic AtoMx export: 2 FOVs of 100 x 100 px side by side, 4 square cells each, 5 genes."""

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

FOVS = {1: (1000, 2100), 2: (1100, 2100)}  # fov: (dx, cy); x_global = dx + col, y_global = cy - row
SQUARES = [(10, 10), (10, 60), (60, 10), (60, 60)]  # (row, col) of each cell's top-left corner, 20 px wide
GENES = ["G1", "G2", "G3", "G4", "G5"]
SLIDE_ID = 1


def _write_export(root, dataset_id="S1"):
    rng = np.random.default_rng(0)
    flat = root / "flatFiles"
    for d in ("flatFiles", "CellComposite", "CellLabels"):
        (root / d).mkdir(parents=True)
    meta, counts, polys, tx = [], [], [], []
    for fov, (dx, cy) in FOVS.items():
        labels = np.zeros((100, 100), np.uint16)
        for k, (r, c) in enumerate(SQUARES, start=1):
            labels[r:r + 20, c:c + 20] = k
            cell = f"c_{SLIDE_ID}_{fov}_{k}"
            meta.append(dict(fov=fov, cell_ID=k, cell=cell, slide_ID=SLIDE_ID, CenterX_local_px=c + 10,
                             CenterY_local_px=r + 10, CenterX_global_px=dx + c + 10, CenterY_global_px=cy - r - 10))
            counts.append(dict(fov=fov, cell_ID=k, **{g: int(v) for g, v in zip(GENES, rng.integers(0, 9, 5))}))
            for rr, cc in [(r, c), (r, c + 20), (r + 20, c + 20), (r + 20, c)]:
                polys.append(dict(fov=fov, cellID=k, cell=cell, x_global_px=dx + cc, y_global_px=cy - rr))
            for g in GENES[:3]:
                tx.append(dict(fov=fov, cell_ID=k, cell=cell, x_global_px=dx + c + 5.0, y_global_px=cy - r - 5.0,
                               z=0, target=g, CellComp="Nuclear"))
        tifffile.imwrite(root / "CellLabels" / f"CellLabels_F{fov:05d}.tif", labels)
        rgb = np.zeros((100, 100, 3), np.uint8)
        rgb[..., fov - 1] = 200  # FOV 1 red, FOV 2 green
        Image.fromarray(rgb).save(root / "CellComposite" / f"CellComposite_F{fov:05d}.jpg", quality=100)
    pd.DataFrame(meta).to_csv(flat / f"{dataset_id}_metadata_file.csv", index=False)
    pd.DataFrame(counts).to_csv(flat / f"{dataset_id}_exprMat_file.csv", index=False)
    pd.DataFrame(polys).to_csv(flat / f"{dataset_id}-polygons.csv", index=False)
    pd.DataFrame(tx).to_csv(flat / f"{dataset_id}_tx_file.csv", index=False)
    return root


@pytest.fixture(scope="session")
def export_dir(tmp_path_factory):
    """<tmp>/S1_inputs, an AtoMx export of dataset S1."""
    return _write_export(tmp_path_factory.mktemp("data") / "S1_inputs")


@pytest.fixture
def sdata(export_dir, tmp_path):
    """The export as SpatialData, with core_str (fov 1: C1R1, fov 2: C2R1) and slide columns."""
    from frzspatial import build_sdata

    s = build_sdata(export_dir, cache_dir=tmp_path / "cache", dataset_id="S1")
    obs = s.tables["table"].obs
    obs["core_str"] = pd.Categorical(np.where(obs["fov"] == 1, "C1R1", "C2R1"))
    obs["slide"] = pd.Categorical(["S1"] * len(obs))
    return s
