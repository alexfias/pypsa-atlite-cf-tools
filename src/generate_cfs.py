#!/usr/bin/env python
"""
Generate wind and PV capacity-factor time series per Voronoi cell using atlite.

- Works with arbitrary cutout files (e.g. different years).
- Lets you set a "run" label so different runs don't overwrite each other.
- Keeps all Voronoi cells (no clipping to cutout bbox).

Example:

    python scripts/generate_cf.py \
        --cutout data/europe-2012-sarah3-era5.nc \
        --voronoi data/bus_voronoi.gpkg \
        --output-dir data/outputs \
        --hub-height 100 \
        --run base \
        --year 2012
"""

import argparse
from pathlib import Path
import re

import atlite
import geopandas as gpd
import shapely
from shapely.geometry import box
from atlite.resource import get_windturbineconfig
import matplotlib.pyplot as plt


def infer_year_from_cutout(path: Path) -> str:
    """
    Try to infer a 4-digit year from the cutout file name.
    Returns "unknown" if no year-like pattern is found.
    """
    m = re.search(r"(19|20)\d{2}", path.name)
    return m.group(0) if m else "unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate wind/PV CF time series per Voronoi cell using atlite."
    )

    parser.add_argument(
        "--cutout",
        required=True,
        type=Path,
        help="Path to atlite cutout NetCDF file (e.g. europe-2012-sarah3-era5.nc).",
    )
    parser.add_argument(
        "--voronoi",
        required=True,
        type=Path,
        help="Path to Voronoi polygons file (e.g. bus_voronoi.gpkg).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/outputs"),
        type=Path,
        help="Directory where CSVs and plots will be written.",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Name of ID column in Voronoi file (e.g. 'bus_id'). If omitted, index is used.",
    )
    parser.add_argument(
        "--hub-height",
        default=100.0,
        type=float,
        help="Wind turbine hub height in meters (default: 100).",
    )
    parser.add_argument(
        "--run",
        default="default",
        help="Run label used in output filenames (e.g. 'base', 's1', 'high-wind').",
    )
    parser.add_argument(
        "--year",
        default=None,
        help=(
            "Year label used in output filenames. If not given, the script tries to "
            "infer it from the cutout file name (e.g. '2012')."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cutout_path: Path = args.cutout
    voronoi_path: Path = args.voronoi
    output_dir: Path = args.output_dir
    id_col = args.id_col
    hub_height = float(args.hub_height)
    run_label = args.run

    year_label = args.year or infer_year_from_cutout(cutout_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Configuration ===")
    print(f"Cutout      : {cutout_path}")
    print(f"Voronoi     : {voronoi_path}")
    print(f"Output dir  : {output_dir}")
    print(f"ID column   : {id_col}")
    print(f"Hub height  : {hub_height} m")
    print(f"Run label   : {run_label}")
    print(f"Year label  : {year_label}")
    print("=====================")

    # ---------- 0) Load inputs ----------
    print("\n=== 0) Load inputs ===")
    cutout = atlite.Cutout(cutout_path)

    gdf_raw = gpd.read_file(voronoi_path)
    print(f"Raw Voronoi cells: {len(gdf_raw)}")
    print(f"Raw CRS: {gdf_raw.crs}")

    # Use ID column if present, otherwise index
    if id_col and id_col in gdf_raw.columns:
        labels_raw = gdf_raw[id_col].astype(str)
        print(f"Using ID column '{id_col}' with {labels_raw.nunique()} unique IDs")
    else:
        labels_raw = gdf_raw.index.astype(str)
        print("No ID_COL used or not found; using index as ID.")

    # Build cutout bbox in 4326 only for plotting / diagnostics
    xmin = float(cutout.data.x.min())
    xmax = float(cutout.data.x.max())
    ymin = float(cutout.data.y.min())
    ymax = float(cutout.data.y.max())
    bbox_4326 = box(xmin, ymin, xmax, ymax)

    # ---------- 1) Reproject & clean geometries (no clipping) ----------
    print("\n=== 1) Reproject & clean geometries (no clipping) ===")
    # Go to 4326 first (if not already)
    gdf_4326 = gdf_raw.to_crs(4326)
    print("After to_crs(4326):", len(gdf_4326))

    # Work in projected CRS for robustness of validity operations
    gdf_proj = gdf_4326.to_crs(3035)
    print("After to_crs(3035):", len(gdf_proj))

    # Make valid
    try:
        from shapely.validation import make_valid as _make_valid
        gdf_proj["geometry"] = gdf_proj.geometry.apply(_make_valid)
        print("Used shapely.validation.make_valid")
    except Exception:
        gdf_proj["geometry"] = gdf_proj.geometry.buffer(0)
        print("Used buffer(0) as make_valid fallback")

    # Snap to 1 m precision grid (removes microscopic self-intersections)
    gdf_proj["geometry"] = gdf_proj.geometry.apply(
        lambda geom: shapely.set_precision(geom, 1.0)
    )

    # Light cleanup: drop truly empty geometries (should be none)
    mask_nonempty = ~gdf_proj.geometry.is_empty
    print("Empty geometries (if any) to drop:", (~mask_nonempty).sum())
    gdf_proj = gdf_proj[mask_nonempty].copy()
    print("After dropping empties:", len(gdf_proj))

    # Back to 4326 for atlite
    gdf_clean_4326 = gdf_proj.to_crs(4326)
    print("After back to CRS=4326:", len(gdf_clean_4326))

    # Ensure we kept the same number of polygons
    if len(gdf_clean_4326) != len(gdf_raw):
        raise RuntimeError(
            f"Lost polygons during cleaning (raw={len(gdf_raw)}, clean={len(gdf_clean_4326)}). "
            "This should not happen in Solution A."
        )

    # Rebuild labels to match geometry order after cleaning
    if id_col and id_col in gdf_raw.columns:
        labels = gdf_raw[id_col].astype(str)
    else:
        labels = gdf_raw.index.astype(str)

    # Align labels with gdf_clean_4326 index
    gdf_clean_4326 = gdf_clean_4326.set_index(gdf_raw.index)
    shapes = gpd.GeoSeries(gdf_clean_4326.geometry.values, index=labels)
    shapes.crs = "EPSG:4326"

    print("\n=== 2) Shapes for atlite ===")
    print("Number of shapes passed to atlite:", len(shapes))
    print("Expected Voronoi cells          :", len(gdf_raw))
    print("First 10 shape labels:", list(shapes.index[:10]))

    # ---------- 3) WIND capacity-factor time series ----------
    print("\n=== 3) atlite.wind ===")
    turb = get_windturbineconfig("Vestas_V112_3MW")
    turb["hub_height"] = hub_height

    cf_wind = cutout.wind(
        turbine=turb,
        shapes=shapes,
        shapes_crs=4326,
        per_unit=True,
        capacity_factor_timeseries=True,
        add_cutout_windspeed=True,
        interpolation_method="logarithmic",
    )

    print("cf_wind dims:", cf_wind.dims)
    print("cf_wind coords:", list(cf_wind.coords))

    if "time" in cf_wind.dims:
        cf_wind = cf_wind.chunk({"time": 240})

    df_wind = cf_wind.to_pandas()

    wind_csv = output_dir / f"cf_wind_by_cell_{year_label}_{run_label}.csv"
    df_wind.to_csv(wind_csv)
    print(f"Wrote {wind_csv} with columns: {len(df_wind.columns)}")

    all_nan_cols = df_wind.columns[df_wind.isna().all()]
    print(
        "Wind: columns that are all-NaN (likely fully outside cutout):",
        len(all_nan_cols),
    )

    # ---------- 4) PV capacity-factor time series ----------
    print("\n=== 4) atlite.pv ===")
    cf_pv = cutout.pv(
        panel="CSi",
        orientation="latitude_optimal",
        tracking=None,
        shapes=shapes,
        shapes_crs=4326,
        per_unit=True,
        capacity_factor_timeseries=True,
    )

    print("cf_pv dims:", cf_pv.dims)
    print("cf_pv coords:", list(cf_pv.coords))

    if "time" in cf_pv.dims:
        cf_pv = cf_pv.chunk({"time": 240})

    df_pv = cf_pv.to_pandas()

    pv_csv = output_dir / f"cf_pv_by_cell_{year_label}_{run_label}.csv"
    df_pv.to_csv(pv_csv)
    print(f"Wrote {pv_csv} with columns: {len(df_pv.columns)}")

    all_nan_cols_pv = df_pv.columns[df_pv.isna().all()]
    print(
        "PV: columns that are all-NaN (likely fully outside cutout):",
        len(all_nan_cols_pv),
    )

    # ---------- 5) Optional quick plot ----------
    try:
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_4326], crs=4326)
        fig, ax = plt.subplots()
        gdf_clean_4326.boundary.plot(ax=ax, linewidth=0.5)
        bbox_gdf.boundary.plot(ax=ax)
        ax.set_title(
            f"Voronoi cells vs cutout area (no clipping, CRS 4326)\n"
            f"year={year_label}, run={run_label}"
        )
        plt.tight_layout()

        plot_path = output_dir / f"voronoi_vs_cutout_no_clip_{year_label}_{run_label}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved {plot_path}")
    except Exception as e:
        print("Plotting failed:", e)

    print("\nDone.")


if __name__ == "__main__":
    main()
