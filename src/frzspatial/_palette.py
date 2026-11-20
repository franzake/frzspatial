"""Category colours: distinct hues, scanpy's defaults, and the colours stored with an obs column."""


def distinct_colours(n):
    """n distinct colours (hex): hues spread by the golden ratio, with three saturation / lightness levels. For more
    categories than scanpy's palettes hold (102), where spatialdata-plot would draw them all grey."""
    import colorsys

    levels = [(0.75, 0.85), (0.55, 0.65), (0.9, 0.55)]  # (saturation, value)
    out = []
    for i in range(n):
        s, v = levels[i % 3]
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1, s, v)
        out.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return out


def scanpy_palette(categories):
    """``{category: colour}`` as scanpy colours a categorical column: default_20, default_28 or default_102 by the
    number of categories, distinct colours beyond 102."""
    from scanpy.pl import palettes

    cats = list(categories)
    n = len(cats)
    cols = palettes.default_20 if n <= 20 else palettes.default_28 if n <= 28 else palettes.default_102 if n <= 102 \
        else distinct_colours(n)
    return dict(zip(cats, cols))


def column_palette(table, column, palette=None):
    """`palette`, else the colours stored with the column (``uns[f"{column}_colors"]``, e.g. by a scanpy UMAP), else
    scanpy's defaults over all its categories (so a value keeps its colour in every plot)."""
    if palette:
        return dict(palette)
    obs = table.obs
    cats = list(obs[column].cat.categories) if hasattr(obs[column], "cat") else sorted(obs[column].dropna().astype(str).unique())
    stored = table.uns.get(f"{column}_colors") if hasattr(table, "uns") else None
    if stored is not None and len(stored) == len(cats):
        return dict(zip(cats, list(stored)))
    return scanpy_palette(cats)
