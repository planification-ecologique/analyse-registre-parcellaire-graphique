import os
from typing import Dict, Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from shapely.geometry import box


def _ensure_epsg_2154(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	"""Project to EPSG:2154 (Lambert-93, meters)."""
	if gdf.crs is None:
		raise ValueError("Input GeoDataFrame has no CRS; please set or provide data with CRS.")
	if str(gdf.crs).upper() in {"EPSG:2154", "EPSG:2154"}:
		return gdf
	return gdf.to_crs("EPSG:2154")


def _grid_from_bounds(bounds: Tuple[float, float, float, float], resolution_m: float) -> Tuple[rasterio.Affine, int, int]:
	"""Create transform and raster shape from bounds and resolution (meters)."""
	minx, miny, maxx, maxy = bounds
	width = int(np.ceil((maxx - minx) / resolution_m))
	height = int(np.ceil((maxy - miny) / resolution_m))
	# Align max to grid
	aligned_maxx = minx + width * resolution_m
	aligned_maxy = miny + height * resolution_m
	transform = from_origin(minx, aligned_maxy, resolution_m, resolution_m)
	return transform, height, width


def rasterize_geojson_cod_cult(
	input_geojson: str,
	output_tiff: str,
	resolution_m: float = 10.0,
	attr_name: str = "cod_cult",
	nodata: int = 0,
	code_mapping: Optional[Dict[str, int]] = None,
	return_mapping_csv: Optional[str] = None,
) -> Dict[str, int]:
	"""
	Rasterize a GeoJSON of parcels to a GeoTIFF labeled by cod_cult.

	- Reprojects to EPSG:2154 (meters).
	- Builds a grid from data extent with given resolution.
	- If values are not integers, a mapping is created to encode them.
	- Returns the mapping dict used. Optionally writes it to CSV (value,int_code).

	Args:
		input_geojson: Path to input GeoJSON file.
		output_tiff: Path to output GeoTIFF file.
		resolution_m: Pixel size in meters.
		attr_name: Attribute containing the culture code.
		nodata: NODATA integer value.
		code_mapping: Optional pre-defined mapping from raw values to int codes.
		return_mapping_csv: Optional CSV path to save mapping table.

	Returns:
		Dict mapping original attribute values (as str) to encoded integer codes.
	"""
	gdf = gpd.read_file(input_geojson)
	if attr_name not in gdf.columns:
		raise KeyError(f"Attribute '{attr_name}' not found in GeoJSON columns: {list(gdf.columns)}")

	gdf = _ensure_epsg_2154(gdf)
	bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
	transform, height, width = _grid_from_bounds(tuple(bounds), resolution_m)

	# Prepare value mapping (ensure integers for raster encoding)
	values = gdf[attr_name].astype(str).fillna("")
	if code_mapping is None:
		unique_vals = sorted(set(values) - {""})
		# start at 1 to keep 0 as nodata
		code_mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}

	shapes: Iterable = (
		(geom, code_mapping.get(val_str, nodata))
		for geom, val_str in zip(gdf.geometry, values.astype(str))
	)

	out_shape = (height, width)
	raster = features.rasterize(
		shapes=shapes,
		out_shape=out_shape,
		transform=transform,
		fill=nodata,
		dtype=np.uint32,
	)

	os.makedirs(os.path.dirname(output_tiff) or ".", exist_ok=True)
	with rasterio.open(
		output_tiff,
		"w",
		driver="GTiff",
		height=height,
		width=width,
		count=1,
		dtype=raster.dtype,
		crs="EPSG:2154",
		transform=transform,
		nodata=nodata,
		compress="deflate",
		predictor=2,
	) as dst:
		dst.write(raster, 1)

	# Optionally save mapping
	if return_mapping_csv:
		pd.DataFrame({attr_name: list(code_mapping.keys()), "code": list(code_mapping.values())}) \
			.sort_values("code").to_csv(return_mapping_csv, index=False)

	return code_mapping


def rasterize_two_geojsons_same_grid(
	geojson_2023: str,
	geojson_2024: str,
	output_tiff_2023: str,
	output_tiff_2024: str,
	resolution_m: float = 10.0,
	attr_name: str = "cod_cult",
	nodata: int = 0,
	return_mapping_csv: Optional[str] = None,
) -> Dict[str, int]:
	"""Rasterize two GeoJSONs onto the same grid/transform, returning the mapping used."""
	gdf23 = _ensure_epsg_2154(gpd.read_file(geojson_2023))
	gdf24 = _ensure_epsg_2154(gpd.read_file(geojson_2024))
	for gdf in (gdf23, gdf24):
		if attr_name not in gdf.columns:
			raise KeyError(f"Attribute '{attr_name}' not found in GeoJSON columns")

	# Union bounds ensures identical grid
	minx = min(gdf23.total_bounds[0], gdf24.total_bounds[0])
	miny = min(gdf23.total_bounds[1], gdf24.total_bounds[1])
	maxx = max(gdf23.total_bounds[2], gdf24.total_bounds[2])
	maxy = max(gdf23.total_bounds[3], gdf24.total_bounds[3])
	transform, height, width = _grid_from_bounds((minx, miny, maxx, maxy), resolution_m)

	# Build shared mapping across both years
	vals = pd.concat([gdf23[attr_name].astype(str), gdf24[attr_name].astype(str)], ignore_index=True).fillna("")
	unique_vals = sorted(set(vals) - {""})
	code_mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}

	def _rasterize(gdf: gpd.GeoDataFrame, out_path: str):
		shapes = ((geom, code_mapping.get(str(v), nodata)) for geom, v in zip(gdf.geometry, gdf[attr_name]))
		r = features.rasterize(
			shapes=shapes,
			out_shape=(height, width),
			transform=transform,
			fill=nodata,
			dtype=np.uint32,
		)
		os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
		with rasterio.open(
			out_path,
			"w",
			driver="GTiff",
			height=height,
			width=width,
			count=1,
			dtype=r.dtype,
			crs="EPSG:2154",
			transform=transform,
			nodata=nodata,
			compress="deflate",
			predictor=2,
		) as dst:
			dst.write(r, 1)

	_rasterize(gdf23, output_tiff_2023)
	_rasterize(gdf24, output_tiff_2024)

	if return_mapping_csv:
		pd.DataFrame({attr_name: list(code_mapping.keys()), "code": list(code_mapping.values())}) \
			.sort_values("code").to_csv(return_mapping_csv, index=False)

	return code_mapping


def compute_transition_matrix_from_rasters(
	raster_2023_path: str,
	raster_2024_path: str,
	output_csv: Optional[str] = None,
	attr_name: str = "cod_cult",
	code_mapping_csv: Optional[str] = None,
) -> pd.DataFrame:
	"""
	Compute the evolution matrix (hectares) between two aligned rasters of cod_cult.

	- Requires identical shape, transform, and CRS.
	- NODATA pixels are ignored.
	- Pixel area is computed from the transform; output values are in hectares.

	Args:
		raster_2023_path: Path to raster for 2023.
		raster_2024_path: Path to raster for 2024.
		output_csv: Optional path to save CSV (wide matrix with rows=2023 codes, cols=2024 codes).
		attr_name: Name of attribute used in mapping CSV (for headers if mapping provided).
		code_mapping_csv: Optional mapping CSV with two columns [attr_name, code] to label rows/cols.

	Returns:
		Pandas DataFrame transition matrix in hectares.
	"""
	with rasterio.open(raster_2023_path) as r23, rasterio.open(raster_2024_path) as r24:
		if (r23.width, r23.height) != (r24.width, r24.height) or r23.transform != r24.transform or r23.crs != r24.crs:
			raise ValueError("Rasters must have identical shape, transform, and CRS.")
		nodata23 = r23.nodata
		nodata24 = r24.nodata
		arr23 = r23.read(1)
		arr24 = r24.read(1)
		transform = r23.transform

	# Pixel area (m^2) then to hectares
	pixel_area_m2 = abs(transform.a) * abs(transform.e)
	pixel_area_ha = pixel_area_m2 / 10000.0

	mask = np.ones(arr23.shape, dtype=bool)
	if nodata23 is not None:
		mask &= arr23 != nodata23
	if nodata24 is not None:
		mask &= arr24 != nodata24

	vals23 = arr23[mask].astype(np.int64)
	vals24 = arr24[mask].astype(np.int64)

	# Build contingency table efficiently
	pairs = pd.DataFrame({"y2023": vals23, "y2024": vals24})
	counts = pairs.value_counts().rename("pixels").reset_index()
	counts["hectares"] = counts["pixels"] * pixel_area_ha

	# Pivot to wide matrix
	matrix = counts.pivot(index="y2023", columns="y2024", values="hectares").fillna(0.0)

	# Replace integer codes with labels if mapping CSV provided
	if code_mapping_csv and os.path.exists(code_mapping_csv):
		mapping_df = pd.read_csv(code_mapping_csv)
		if {attr_name, "code"}.issubset(mapping_df.columns):
			label_by_code = mapping_df.set_index("code")[attr_name].to_dict()
			matrix.index = [label_by_code.get(int(c), c) for c in matrix.index]
			matrix.columns = [label_by_code.get(int(c), c) for c in matrix.columns]

	if output_csv:
		os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
		matrix.to_csv(output_csv, float_format="%.4f")

	return matrix


def export_transition_matrix_csv(matrix: pd.DataFrame, output_csv: str) -> None:
	"""Export the provided transition matrix to CSV."""
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
	matrix.to_csv(output_csv, float_format="%.4f")


__all__ = [
	"rasterize_geojson_cod_cult",
	"rasterize_two_geojsons_same_grid",
	"compute_transition_matrix_from_rasters",
	"export_transition_matrix_csv",
]


