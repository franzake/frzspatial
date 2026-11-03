"""Command line: ``frzspatial build``, ``frzspatial info`` and ``frzspatial view``. See ``frzspatial --help``."""

import argparse
import logging
import sys
from pathlib import Path


def _build(args):
    from .io import build_sdata

    out = Path(args.out or f"{args.dataset_id}_flat.zarr")
    sdata = build_sdata(args.inputs, cache_dir=args.cache_dir, dataset_id=args.dataset_id,
                        morphology=args.morphology, transcripts=not args.no_transcripts, threads=args.threads)
    sdata.write(out, overwrite=args.overwrite)
    print(f"written {out}")


def _info(args):
    logging.getLogger("ome_zarr.reader").setLevel(logging.ERROR)
    from spatialdata import read_zarr

    print(read_zarr(args.zarr))


def _view(args):
    logging.getLogger("ome_zarr.reader").setLevel(logging.ERROR)
    from spatialdata import read_zarr

    from .viewer import annotate_core, open_napari

    sdata = read_zarr(args.zarr)
    if args.annotate:
        if not args.slide:
            sys.exit("--annotate needs --slide")
        annotate_core(sdata, args.slide, args.core, args.annotate, annot_dir=args.annot_dir)
    elif args.core:
        from .core import build_core

        open_napari(build_core(sdata, args.slide, args.core))
    else:
        open_napari(sdata)


def main(argv=None):
    p = argparse.ArgumentParser(prog="frzspatial", description="CosMx (AtoMx) exports to SpatialData.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a SpatialData zarr from an AtoMx export folder")
    b.add_argument("inputs", help="export folder with flatFiles/, CellComposite/, CellLabels/")
    b.add_argument("dataset_id", help="prefix of the flat files, usually the slide name")
    b.add_argument("-o", "--out", help="output zarr (default: <dataset_id>_flat.zarr)")
    b.add_argument("--cache-dir", default="cosmx_cache", help="transcript Parquet cache (default: cosmx_cache)")
    b.add_argument("--morphology", action="store_true", help="also read Morphology2D/")
    b.add_argument("--no-transcripts", action="store_true", help="skip the transcripts")
    b.add_argument("--threads", type=int, default=8)
    b.add_argument("--overwrite", action="store_true")
    b.set_defaults(func=_build)

    i = sub.add_parser("info", help="print the elements of a SpatialData zarr")
    i.add_argument("zarr")
    i.set_defaults(func=_info)

    v = sub.add_parser("view", help="open a SpatialData zarr (or one of its cores) in napari")
    v.add_argument("zarr")
    v.add_argument("--slide", help="slide name (needed with --core or --annotate)")
    v.add_argument("--core", help="stitch and show one core (obs core_str)")
    v.add_argument("--annotate", nargs="+", metavar="CLASS", help="draw annotations of these classes")
    v.add_argument("--annot-dir", default="results/annotations")
    v.set_defaults(func=_view)

    args = p.parse_args(argv)
    if args.command == "view" and args.core and not args.slide:
        p.error("--core needs --slide")
    args.func(args)


if __name__ == "__main__":
    main()
