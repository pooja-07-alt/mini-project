"""
preprocessing.py
────────────────
Geospatial preprocessing utilities.

Stage 3 will use these functions to:
  - Clip rasters to the study area boundary
  - Reproject to UTM Zone 43N (EPSG:32643)
  - Resample all rasters to a common 30m grid
  - Align extents

All functions follow the same signature pattern:
    process_xxx(input_path, output_path, **kwargs) -> Path
"""

from pathlib import Path
from typing import Union

# NOTE: These imports will fail until you install the packages.
# Run: pip install rasterio geopandas shapely
try:
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask
    import geopandas as gpd
    import numpy as np
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False
    print("⚠️  rasterio / geopandas not installed. Run: pip install rasterio geopandas")

# Target CRS: UTM Zone 43N — metric units, correct for Karnataka
TARGET_CRS = "EPSG:32643"
TARGET_RESOLUTION_M = 30  # 30-metre grid


def clip_raster_to_boundary(
    raster_path: Union[str, Path],
    boundary_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """
    Clip a raster to a vector boundary (e.g. sub-watershed polygon).

    Parameters
    ----------
    raster_path   : Path to input raster (.tif)
    boundary_path : Path to boundary shapefile or GeoJSON
    output_path   : Path for output clipped raster

    Returns
    -------
    Path to output file
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio and geopandas must be installed.")

    boundary = gpd.read_file(boundary_path)
    with rasterio.open(raster_path) as src:
        # Reproject boundary to match raster CRS before masking
        boundary = boundary.to_crs(src.crs)
        shapes = [geom.__geo_interface__ for geom in boundary.geometry]
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width":  out_image.shape[2],
            "transform": out_transform,
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"✅  Clipped raster saved to: {output_path}")
    return output_path


def reproject_raster(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_crs: str = TARGET_CRS,
    resolution_m: float = TARGET_RESOLUTION_M,
    resampling_method: str = "bilinear",
) -> Path:
    """
    Reproject a raster to the target CRS and resolution.

    Parameters
    ----------
    input_path      : Path to input raster
    output_path     : Path for output reprojected raster
    target_crs      : Target CRS string (default: UTM 43N)
    resolution_m    : Target resolution in metres (default: 30)
    resampling_method : One of 'bilinear', 'nearest', 'cubic'

    Returns
    -------
    Path to output file
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio must be installed.")

    resampling_map = {
        "bilinear": Resampling.bilinear,
        "nearest":  Resampling.nearest,
        "cubic":    Resampling.cubic,
    }
    resampling = resampling_map.get(resampling_method, Resampling.bilinear)

    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds,
            resolution=(resolution_m, resolution_m),
        )
        meta = src.meta.copy()
        meta.update({
            "crs":       target_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
        })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    print(f"✅  Reprojected raster saved to: {output_path}")
    return output_path


def align_rasters_to_reference(
    raster_paths: list,
    reference_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> list:
    """
    Align multiple rasters to the same grid as a reference raster.

    Useful for ensuring DEM, LULC, rainfall, etc. all have exactly
    the same extent, resolution, and CRS before building the feature table.

    Parameters
    ----------
    raster_paths   : List of paths to rasters that need aligning
    reference_path : Path to the reference raster (usually the DEM)
    output_dir     : Directory for aligned output rasters

    Returns
    -------
    List of paths to aligned output rasters
    """
    if not _RASTERIO_AVAILABLE:
        raise ImportError("rasterio must be installed.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height

    output_paths = []
    for raster_path in raster_paths:
        raster_path = Path(raster_path)
        output_path = output_dir / f"{raster_path.stem}_aligned.tif"

        with rasterio.open(raster_path) as src:
            meta = src.meta.copy()
            meta.update({
                "crs":       ref_crs,
                "transform": ref_transform,
                "width":     ref_width,
                "height":    ref_height,
            })
            with rasterio.open(output_path, "w", **meta) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )

        print(f"✅  Aligned: {raster_path.name} → {output_path.name}")
        output_paths.append(output_path)

    return output_paths
