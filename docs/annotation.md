# Annotation in napari

1. Open the slide and make sure its table has the `slide` and `core_str` obs columns.
2. Run `annotate_core(sdata, slide, core, classes)` (`core=None` for the whole slide, at 1/8 resolution). napari
   opens with the core's composite, its cell outlines and one empty layer `annot: <class>` per class. Shapes saved
   for this core earlier are loaded back.
3. Click the layer of a class and draw: **P** for a polygon (double-click to close), **R** for a rectangle (a
   small one over a cell's centre marks a single cell), **5** to select, move or delete. Ellipses are ignored.
4. Every change is saved to `<annot_dir>/<slide>_<core>.geojson`, in the slide's global px.
5. Close the window with its own close button (not Cmd-Q when Python runs inside R or Jupyter). The polygon count
   is printed and `obs["manual_annotation"]` is rebuilt from every file in `annot_dir`.
6. Check it with `obs["manual_annotation"].value_counts()` or
   `plot_core(sdata, core, color="manual_annotation", palette=annotation_palette(classes, obs["manual_annotation"]))`.
7. Store the column in the slide's zarr with `save_table(sdata)`.

Each cell takes the class of the polygon that contains its centroid; where polygons overlap, the class later in
`classes` wins. The column is categorical with every class as a category, so tables annotated separately still
share their categories when they are concatenated.

If napari hangs or the host process dies, nothing drawn is lost: rerun `annotate_core` to reopen the shapes, or
`annotations_to_obs(sdata, classes)` to rebuild the column.
