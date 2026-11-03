import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from spatialdata.transformations import get_transformation

import frzspatial as fs
from frzspatial.cli import main


def test_build_sdata(sdata):
    assert set(sdata.images) == {"fov1_image", "fov2_image"}
    assert set(sdata.labels) == {"fov1_labels", "fov2_labels"}
    table = sdata.tables["table"]
    assert table.shape == (8, 5)
    assert len(sdata.shapes["cell_boundaries"]) == 8
    assert len(sdata.points["transcripts"].compute()) == 24
    m = get_transformation(sdata.images["fov2_image"], "global").to_affine_matrix(("x", "y"), ("x", "y"))
    assert (m[0, 2], m[1, 2], m[1, 1]) == (1100, 2100, -1)  # pixel (0, 0) at (dx, cy), y flipped
    assert "slide" in sdata.coordinate_systems


def test_polygons_contain_centroids(sdata):
    obs = sdata.tables["table"].obs
    shp = sdata.shapes["cell_boundaries"].loc[obs.index]
    assert shapely.contains_xy(shp.geometry.to_numpy(), obs["CenterX_global_px"], obs["CenterY_global_px"]).all()


def test_load_slides_and_save_table(export_dir, tmp_path):
    zarr_dir = tmp_path / "zarr"
    slides = fs.load_slides(["S1"], input_dir=export_dir.parent, zarr_dir=zarr_dir, cache_dir=tmp_path / "cache")
    assert (zarr_dir / "S1_flat.zarr").exists()
    s = slides["S1"]
    s.tables["table"].obs["flag"] = "x"
    fs.save_table(s)
    again = fs.load_slides(["S1"], input_dir=export_dir.parent, zarr_dir=zarr_dir, cache_dir=tmp_path / "cache")
    assert "flag" in again["S1"].tables["table"].obs


def test_load_slides_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        fs.load_slides(["nope"], input_dir=tmp_path, zarr_dir=tmp_path / "zarr", cache_dir=tmp_path)


def test_build_core(sdata):
    core = fs.build_core(sdata, "S1", "C1R1")
    assert core["core_image"]["scale0"]["image"].shape == (3, 100, 100)
    labels = np.asarray(core["core_labels"]["scale0"]["image"])
    table = core.tables["table"]
    # every cell's mask value is its id in the table and the polygons
    assert set(np.unique(labels)) - {0} == set(table.obs["cell"])
    assert set(core.shapes["cell_boundaries"].index) == set(table.obs["cell"])
    assert set(table.obs["core_str"].cat.categories) <= {"C1R1", "C2R1"}


def test_build_core_whole_slide(sdata):
    slide = fs.build_core(sdata, "S1", None, downsample=2)
    assert slide["core_image"]["scale0"]["image"].shape == (3, 50, 100)  # two FOVs side by side, half resolution
    m = get_transformation(slide["core_image"], "global").to_affine_matrix(("x", "y"), ("x", "y"))
    assert (m[0, 0], m[0, 2], m[1, 2]) == (2, 1000, 2100)


def test_build_core_unknown(sdata):
    with pytest.raises(ValueError):
        fs.build_core(sdata, "S1", "C9R9")


def test_annotations_to_obs(sdata, tmp_path):
    # FOV 1 cells: centres at x 1020/1070, y 2080/2030. A box over the top two, another over the top-left one
    box = shapely.box(1000, 2060, 1100, 2100)
    small = shapely.box(1010, 2070, 1030, 2090)
    annot = gpd.GeoDataFrame({"class": ["Glomerulus", "Artery"]}, geometry=[box, small])
    fs.write_annotations(annot, fs.annotation_file(tmp_path, "S1", "C1R1"))
    fs.write_annotations(annot, fs.annotation_file(tmp_path, "OTHER", "C1R1"))  # another slide: skipped

    counts = fs.annotations_to_obs(sdata, ["Glomerulus", "Artery", "Exclude"], tmp_path)
    assert counts.to_dict() == {"Glomerulus": 1, "Artery": 1, "Exclude": 0}  # later class wins on overlap
    col = sdata.tables["table"].obs["manual_annotation"]
    assert list(col.cat.categories) == ["Glomerulus", "Artery", "Exclude"]
    assert col.isna().sum() == 6

    fs.write_annotations(annot.iloc[:0], fs.annotation_file(tmp_path, "S1", "C1R1"))  # empty: file deleted
    assert not fs.annotation_file(tmp_path, "S1", "C1R1").exists()


def test_whole_slide_annotation(sdata, tmp_path):
    fs.write_annotations(gpd.GeoDataFrame({"class": ["Inflammation"]}, geometry=[shapely.box(0, 0, 5000, 5000)]),
                         fs.annotation_file(tmp_path, "S1"))
    counts = fs.annotations_to_obs(sdata, ["Inflammation"], tmp_path)
    assert counts["Inflammation"] == 8


def test_write_seurat_inputs(sdata, tmp_path):
    t = sdata.tables["table"]
    out = fs.write_seurat_inputs(tmp_path / "seurat", {"RNA": t.X}, t.var_names, t.obs_names, t.obs[["fov"]],
                                 t.obsm["spatial"])
    i = np.fromfile(out / "RNA_i.bin", np.int32)
    p = np.fromfile(out / "RNA_p.bin", np.int32)
    x = np.fromfile(out / "RNA_x.bin", np.float64)
    import scipy.sparse as sp

    back = sp.csc_matrix((x, i, p), shape=(t.n_vars, t.n_obs))
    assert (back.T != t.X).nnz == 0
    assert pd.read_csv(out / "metadata.csv", index_col=0).shape == (8, 1)


def test_cli_build_and_info(export_dir, tmp_path, capsys):
    out = tmp_path / "S1.zarr"
    main(["build", str(export_dir), "S1", "-o", str(out), "--cache-dir", str(tmp_path / "cache")])
    assert out.exists()
    main(["info", str(out)])
    assert "cell_boundaries" in capsys.readouterr().out


def test_plots(sdata):
    pytest.importorskip("spatialdata_plot")
    import matplotlib

    matplotlib.use("Agg")
    from frzspatial.pl import annotation_palette, plot_core, plot_slide_overview

    sdata.tables["table"].obs["manual_annotation"] = pd.Categorical(["Glomerulus"] * 2 + [None] * 6,
                                                                    categories=["Glomerulus", "Artery"])
    pal = annotation_palette(["Glomerulus", "Artery"], sdata.tables["table"].obs["manual_annotation"])
    assert pal == {"Glomerulus": "#1f77b4"}
    assert plot_core(sdata, "C1R1", genes={"G1": "magenta"}) is not None
    assert plot_core(sdata, "C1R1", color="manual_annotation", palette=pal) is not None
    assert plot_slide_overview(sdata, "core_str", dpi=50) is not None
