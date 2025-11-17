import atlite
import geopandas as gpd
import shapely
from shapely.geometry import box
from atlite.resource import get_windturbineconfig

import matplotlib.pyplot as plt

# ---------- INPUTS ----------
CUTOUT_PATH = "europe-2012-sarah3-era5.nc"   # downloaded from Zenodo
VORONOI_PATH = "./bus_voronoi.gpkg"          # your Voronoi file
ID_COL = None                               # e.g. "bus_id" if you add one later
HUB_HEIGHT_M = 100                           # wind hub height
# ----------------------------


def main():
    print("=== 0) Load inputs ===")
    cutout = atlite.Cutout(CUTOUT_PATH)

    gdf_raw = gpd.read_file(VORONOI_PATH)
    print(f"Raw Voronoi cells: {len(gdf_raw)}")
    print(f"Raw CRS: {gdf_raw.crs}")

    # Use ID column if present, otherwise index
    if ID_COL and ID_COL in gdf_raw.columns:
        labels_raw = gdf_raw[ID_COL].astype(str)
        print(f"Using ID column '{ID_COL}' with {labels_raw.nunique()} unique IDs")
    else:
        labels_raw = gdf_raw.index.astype(str)
        print("No ID column used; using index as ID.")

    # Build cutout bbox in 4326 just for plotting / diagnostics (NOT for clipping)
    xmin = float(cutout.data.x.min()); xmax = float(cutout.data.x.max())
    ymin = float(cutout.data.y.min()); ymax = float(cutout.data.y.max())
    bbox_4326 = box(xmin, ymin, xmax, ymax)

    # --- 1) Reproject Voronoi cells to WGS84 ---
    print("\n=== 1) Reproject & clean geometries (no clipping) ===")
    # Your GPKG is in EPSG:3035; go to 4326 first
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

    # Light cleanup: drop only truly empty geometries (should be none)
    mask_nonempty = ~gdf_proj.geometry.is_empty
    print("Empty geometries (if any) to drop:", (~mask_nonempty).sum())
    gdf_proj = gdf_proj[mask_nonempty].copy()
    print("After dropping empties:", len(gdf_proj))

    # Back to 4326 for atlite
    gdf_clean_4326 = gdf_proj.to_crs(4326)
    print("After back to CRS=4326:", len(gdf_clean_4326))

    # Make sure we still have one shape per original cell
    assert len(gdf_clean_4326) == len(gdf_raw), \
        "Lost polygons during cleaning – this should not happen in Solution A."

    # Rebuild labels to match geometry order after cleaning
    if ID_COL and ID_COL in gdf_raw.columns:
        # Keep order from gdf_raw (same index positions)
        labels = gdf_raw[ID_COL].astype(str)
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

    # --- 3) WIND capacity-factor time series per Voronoi cell ---
    print("\n=== 3) atlite.wind (no clipping) ===")
    turb = get_windturbineconfig("Vestas_V112_3MW")
    turb["hub_height"] = HUB_HEIGHT_M

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
    df_wind.to_csv("cf_wind_by_cell_2012.csv")
    print("Wrote cf_wind_by_cell_2012.csv with columns:", len(df_wind.columns))

    # Simple NaN diagnostics
    all_nan_cols = df_wind.columns[df_wind.isna().all()]
    print("Wind: columns that are all-NaN (likely fully outside cutout):",
          len(all_nan_cols))

    # --- 4) PV capacity-factor time series per Voronoi cell ---
    print("\n=== 4) atlite.pv (no clipping) ===")
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
    df_pv.to_csv("cf_pv_by_cell_2012.csv")
    print("Wrote cf_pv_by_cell_2012.csv with columns:", len(df_pv.columns))

    all_nan_cols_pv = df_pv.columns[df_pv.isna().all()]
    print("PV: columns that are all-NaN (likely fully outside cutout):",
          len(all_nan_cols_pv))

    # --- 5) Optional quick plot: Voronoi vs cutout bbox (just to eyeball) ---
    try:
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_4326], crs=4326)
        ax = gdf_clean_4326.boundary.plot(linewidth=0.5)
        bbox_gdf.boundary.plot(ax=ax)
        ax.set_title("Voronoi cells vs cutout area (no clipping, CRS 4326)")
        plt.tight_layout()
        plt.savefig("voronoi_vs_cutout_no_clip.png", dpi=150)
        print("Saved voronoi_vs_cutout_no_clip.png")
    except Exception as e:
        print("Plotting failed:", e)

    print("\nDone. All original Voronoi cells were kept; some may have all-NaN time series if fully outside the cutout.")

