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
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import fiona


def _iter_with_progress(iterable: Iterable, total: Optional[int] = None, desc: Optional[str] = None) -> Iterable:
	"""Yield items from iterable while displaying a progress bar if tqdm is available."""
	try:
		from tqdm import tqdm  # type: ignore
		return tqdm(iterable, total=total, desc=desc)
	except Exception:
		# Fallback: no progress bar
		return iterable


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
	attr_name: str = "CODE_CULTU",
	nodata: int = 0,
	code_mapping: Optional[Dict[str, int]] = None,
	return_mapping_csv: Optional[str] = None,
) -> Dict[str, int]:
	"""
	Rasterize a GeoJSON of parcels to a GeoTIFF labeled by CODE_CULTU.

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

	# Wrap feature iteration with progress
	shapes: Iterable = (
		(geom, code_mapping.get(val_str, nodata))
		for geom, val_str in _iter_with_progress(
			zip(gdf.geometry, values.astype(str)),
			total=len(gdf),
			desc="Rasterizing"
		)
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
	attr_name: str = "CODE_CULTU",
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
		shapes = (
			(geom, code_mapping.get(str(v), nodata))
			for geom, v in _iter_with_progress(
				zip(gdf.geometry, gdf[attr_name]), total=len(gdf), desc="Rasterizing"
			)
		)
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
	attr_name: str = "CODE_CULTU",
	code_mapping_csv: Optional[str] = None,
) -> pd.DataFrame:
	"""
	Compute the evolution matrix (hectares) between two aligned rasters of CODE_CULTU.

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

		# Include pixels where exactly one year has data. Exclude only pixels where both are NODATA.
		if nodata23 is None and nodata24 is None:
			mask = np.ones(arr23.shape, dtype=bool)
		else:
			cond23 = (arr23 == nodata23) if nodata23 is not None else np.zeros(arr23.shape, dtype=bool)
			cond24 = (arr24 == nodata24) if nodata24 is not None else np.zeros(arr24.shape, dtype=bool)
			# keep where not (both nodata)
			mask = ~(cond23 & cond24)

		vals23 = arr23[mask].astype(np.int64)
		vals24 = arr24[mask].astype(np.int64)

	# Build contingency table efficiently
	pairs = pd.DataFrame({"y2023": vals23, "y2024": vals24})
	counts = pairs.value_counts().rename("pixels").reset_index()
	counts["hectares"] = counts["pixels"] * pixel_area_ha

	# Pivot to wide matrix
	matrix = counts.pivot(index="y2023", columns="y2024", values="hectares").fillna(0.0)

	# Replace integer codes with labels; map 0 to "NO CULTURE"
	label_by_code: Dict[int, str] = {}
	if code_mapping_csv and os.path.exists(code_mapping_csv):
		mapping_df = pd.read_csv(code_mapping_csv)
		if {attr_name, "code"}.issubset(mapping_df.columns):
			label_by_code = mapping_df.set_index("code")[attr_name].to_dict()

	def _label_for(code: int):
		try:
			code_int = int(code)
		except Exception:
			return code
		if code_int == 0:
			return "NO CULTURE"
		return label_by_code.get(code_int, code)

	matrix.index = [_label_for(c) for c in matrix.index]
	matrix.columns = [_label_for(c) for c in matrix.columns]

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


def rasterize_geojson_cod_cult_streaming(
	input_geojson: str,
	output_tiff: str,
	resolution_m: float = 10.0,
	attr_name: str = "CODE_CULTU",
	nodata: int = 0,
	code_mapping: Optional[Dict[str, int]] = None,
	return_mapping_csv: Optional[str] = None,
	batch_size: int = 500,
) -> Dict[str, int]:
	"""
	Rasterize a potentially very large GeoJSON by streaming features with Fiona.

	- Reprojects geometries to EPSG:2154 on the fly.
	- Uses a batching strategy to avoid holding all shapes in memory.
	- If no mapping provided, builds it in a first streaming pass.
	"""
	# Open source and get bounds/crs cheaply
	with fiona.open(input_geojson, "r") as src:
		src_crs = src.crs_wkt or src.crs
		minx, miny, maxx, maxy = src.bounds

	# Prepare transformer to EPSG:2154
	transformer = Transformer.from_crs(src_crs, "EPSG:2154", always_xy=True)

	# Transform bounds to EPSG:2154 for grid creation
	minx2154, miny2154 = transformer.transform(minx, miny)
	maxx2154, maxy2154 = transformer.transform(maxx, maxy)
	transform, height, width = _grid_from_bounds((minx2154, miny2154, maxx2154, maxy2154), resolution_m)

	# First pass to build mapping if needed
	if code_mapping is None:
		unique_vals: set[str] = set()
		with fiona.open(input_geojson, "r") as src:
			for feat in _iter_with_progress(src, total=None, desc="Scanning codes"):
				val = str(feat.get("properties", {}).get(attr_name, ""))
				if val:
					unique_vals.add(val)
		code_mapping = {val: idx + 1 for idx, val in enumerate(sorted(unique_vals))}

	# Rasterize in batches, writing to array
	arr = np.full((height, width), nodata, dtype=np.uint32)

	def _reproject_geom(geom: dict) -> object:
		from shapely.geometry import shape
		shp = shape(geom)
		return shapely_transform(lambda x, y: transformer.transform(x, y), shp)

	batch: list[tuple] = []
	with fiona.open(input_geojson, "r") as src:
		it = _iter_with_progress(src, total=None, desc="Rasterizing")
		for feat in it:
			props = feat.get("properties", {})
			val = str(props.get(attr_name, ""))
			code = code_mapping.get(val, nodata)
			geom_proj = _reproject_geom(feat["geometry"]) if feat.get("geometry") else None
			if geom_proj is None:
				continue
			batch.append((geom_proj, code))
			if len(batch) >= batch_size:
				burn = features.rasterize(batch, out_shape=(height, width), transform=transform, fill=nodata, dtype=arr.dtype)
				mask = burn != nodata
				arr[mask] = burn[mask]
				batch.clear()
		# flush remaining
		if batch:
			burn = features.rasterize(batch, out_shape=(height, width), transform=transform, fill=nodata, dtype=arr.dtype)
			mask = burn != nodata
			arr[mask] = burn[mask]

	# Write GeoTIFF
	os.makedirs(os.path.dirname(output_tiff) or ".", exist_ok=True)
	with rasterio.open(
		output_tiff,
		"w",
		driver="GTiff",
		height=height,
		width=width,
		count=1,
		dtype=arr.dtype,
		crs="EPSG:2154",
		transform=transform,
		nodata=nodata,
		compress="deflate",
		predictor=2,
	) as dst:
		dst.write(arr, 1)

	# Optionally save mapping
	if return_mapping_csv:
		pd.DataFrame({attr_name: list(code_mapping.keys()), "code": list(code_mapping.values())}) \
			.sort_values("code").to_csv(return_mapping_csv, index=False)

	return code_mapping
