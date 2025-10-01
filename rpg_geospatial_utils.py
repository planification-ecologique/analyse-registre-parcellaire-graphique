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
import json



# Approximate EPSG:2154 bounds for hexagonal France (Lambert-93)
FR_HEX_BOUNDS_2154: Tuple[float, float, float, float] = (0.0, 6000000.0, 1200000.0, 7200000.0)

# Use fixed mapping if available and hexagonal France bounds to skip scanning
fixed_mapping_csv = os.path.join(os.path.dirname(__file__), "data", "code_cultu_map.csv")
df_map = pd.read_csv(fixed_mapping_csv, sep=';')
code_mapping = {row['code_culture']: idx + 1 for idx, row in df_map.iterrows()}
code_mapping['NO CULTURE'] = 0

print(f"Loaded {len(code_mapping)} culture codes from {fixed_mapping_csv}")

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



def count_features(file_path: str) -> int:
	"""Count the number of line-delimited GeoJSON Feature lines in a file."""
	count = 0
	with open(file_path, "r") as f:
		for line in f:
			if line.strip().startswith('{ "type": "Feature"'):
				count += 1
	return count


def rasterize_geojson_cod_cult(
	input_geojson: str,
	output_tiff: str,
	resolution_m: float = 10.0,
	attr_name: str = "CODE_CULTU",
	nodata: int = 0,
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


def rasterize_geojson_cod_cult_streaming(
	input_geojson: str,
	output_tiff: str,
	resolution_m: float = 10.0,
	attr_name: str = "CODE_CULTU",
	nodata: int = 0,
	return_mapping_csv: Optional[str] = None,
	batch_size: int = 500,
	code_mapping_csv: Optional[str] = None,
) -> Dict[str, int]:
	"""
	Rasterize a potentially very large GeoJSON by streaming features line-by-line.

	- Assumes GeoJSON coordinates are in EPSG:4326; reprojects to EPSG:2154 on the fly.
	- Uses a batching strategy to avoid holding all shapes in memory.
	- If no mapping provided, builds it in a first streaming pass.
	"""
	print(f"[streaming-single] Start — input: {input_geojson}")
	# Inputs are already in EPSG:2154; no reprojection needed
	transformer = None

	# Optional: load code mapping from CSV
	if code_mapping_csv and os.path.exists(code_mapping_csv):
		try:
			df_map = pd.read_csv(code_mapping_csv)
			if {attr_name, "code"}.issubset(df_map.columns):
				code_mapping = df_map.set_index(attr_name)["code"].astype(int).to_dict()
				print(f"[streaming-single] Loaded mapping from {code_mapping_csv} ({len(code_mapping)} codes)")
		except Exception as e:
			print(f"[streaming-single] Warning: failed to load mapping CSV: {e}")

	print("[streaming-single] First pass — scanning bounds...")
	minx2154 = float("inf"); miny2154 = float("inf"); maxx2154 = float("-inf"); maxy2154 = float("-inf")
	feature_count = 0
	try:
		from tqdm import tqdm  # type: ignore
		total_first = count_features(input_geojson)
		it = tqdm(open(input_geojson, "r"), total=total_first if total_first > 0 else None, desc="Scanning bounds")
	except Exception:
		it = open(input_geojson, "r")
	for line in it:
		line = line.strip()
		if not line.startswith('{ "type": "Feature"'):
			continue
		try:
			feat = json.loads(line.rstrip(','))
		except Exception:
			continue
		feature_count += 1
		try:
			from shapely.geometry import shape
			geom = shape(feat.get("geometry")) if feat.get("geometry") else None
			if geom is None:
				continue
			gxmin, gymin, gxmax, gymax = geom.bounds
			minx2154 = min(minx2154, gxmin); miny2154 = min(miny2154, gymin)
			maxx2154 = max(maxx2154, gxmax); maxy2154 = max(maxy2154, gymax)
		except Exception:
			continue
	try:
		it.close()
	except Exception:
		pass
	

	if code_mapping is None:
		code_mapping = {val: idx + 1 for idx, val in enumerate(sorted(unique_vals))}

	print("[streaming-single] Building grid...")
	transform, height, width = _grid_from_bounds((minx2154, miny2154, maxx2154, maxy2154), resolution_m)

	print("[streaming-single] Second pass — rasterizing features...")
	arr = np.full((height, width), nodata, dtype=np.uint32)
	batch: list[tuple] = []
	try:
		from tqdm import tqdm  # type: ignore
		progress_iter = tqdm(desc="Rasterizing", total=(feature_count if (feature_count > 0 and not use_hex_france_bounds) else None))
		use_tqdm = True
	except Exception:
		progress_iter = None
		use_tqdm = False

	with open(input_geojson, "r") as f:
		for line in f:
			line = line.strip()
			if not line.startswith('{ "type": "Feature"'):
				continue
			try:
				feat = json.loads(line.rstrip(','))
			except Exception:
				if use_tqdm:
					progress_iter.update(1)
				continue
			props = feat.get("properties", {})
			val = str(props.get(attr_name, ""))
			code = code_mapping.get(val, nodata)
			try:
				from shapely.geometry import shape
				geom = shape(feat.get("geometry")) if feat.get("geometry") else None
				if geom is None:
					if use_tqdm:
						progress_iter.update(1)
					continue
				geom_2154 = geom
			except Exception:
				if use_tqdm:
					progress_iter.update(1)
				continue
			batch.append((geom_2154, code))
			if len(batch) >= batch_size:
				burn = features.rasterize(batch, out_shape=(height, width), transform=transform, fill=nodata, dtype=arr.dtype)
				mask = burn != nodata
				arr[mask] = burn[mask]
				batch.clear()
			if use_tqdm:
				progress_iter.update(1)
		# flush remaining
		if batch:
			burn = features.rasterize(batch, out_shape=(height, width), transform=transform, fill=nodata, dtype=arr.dtype)
			mask = burn != nodata
			arr[mask] = burn[mask]
	if use_tqdm:
		progress_iter.close()

	print(f"[streaming-single] Writing GeoTIFF → {output_tiff}")
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
		print(f"[streaming-single] Saving mapping CSV → {return_mapping_csv}")
		pd.DataFrame({attr_name: list(code_mapping.keys()), "code": list(code_mapping.values())}) \
			.sort_values("code").to_csv(return_mapping_csv, index=False)

	print("[streaming-single] Done.")
	return code_mapping


def rasterize_two_geojsons_tiled(
	geojson_2023: str,
	geojson_2024: str,
	output_tiff_2023: str,
	output_tiff_2024: str,
	resolution_m: float = 10.0,
	attr_name: str = "CODE_CULTU",
	nodata: int = 0,
	return_mapping_csv: Optional[str] = None,
	tiles_x: int = 10,
	tiles_y: int = 10,
) -> Dict[str, int]:
	"""
	Rasterize two GeoJSONs using tiled approach with temporary files.
	
	- Uses fixed hexagonal France bounds divided into tiles.
	- Creates temporary tile files, processes features into them.
	- Combines tiles into final output rasters at the end.
	"""
	import tempfile
	import shutil
	from pathlib import Path
	
	print(f"[tiled] Start — inputs: {geojson_2023}, {geojson_2024}")
	
	# Use fixed hexagonal France bounds
	minx, miny, maxx, maxy = FR_HEX_BOUNDS_2154
	print(f"[tiled] Using hexagonal France bounds: {minx}, {miny}, {maxx}, {maxy}")
	
	# Calculate full grid dimensions
	transform, height, width = _grid_from_bounds((minx, miny, maxx, maxy), resolution_m)
	print(f"[tiled] Full grid: {width}x{height} pixels")
	
	# Calculate tile dimensions
	tile_width = width // tiles_x
	tile_height = height // tiles_y
	print(f"[tiled] Tile size: {tile_width}x{tile_height} pixels ({tiles_x}x{tiles_y} tiles)")

	# Create temporary directory for tile files
	temp_dir = Path(tempfile.mkdtemp(prefix="rpg_tiles_"))
	print(f"[tiled] Using temporary directory: {temp_dir}")
	
	try:
		# Create temporary tile files for each year
		tile_files_2023 = {}
		tile_files_2024 = {}
		
		for ty in range(tiles_y):
			for tx in range(tiles_x):
				col_off = tx * tile_width
				row_off = ty * tile_height
				col_size = min(tile_width, width - col_off)
				row_size = min(tile_height, height - row_off)
				
				# Create tile window and transform
				tile_window = rasterio.windows.Window(col_off, row_off, col_size, row_size)
				tile_transform = rasterio.windows.transform(tile_window, transform)
				
				# Create temporary tile files
				tile_file_2023 = temp_dir / f"tile_2023_{tx}_{ty}.tif"
				tile_file_2024 = temp_dir / f"tile_2024_{tx}_{ty}.tif"
				
				# Initialize empty tile files
				with rasterio.open(
					tile_file_2023, 'w',
					driver='GTiff',
					height=row_size,
					width=col_size,
					count=1,
					dtype=np.uint32,
					crs='EPSG:2154',
					transform=tile_transform,
					nodata=nodata,
				) as dst:
					dst.write(np.full((row_size, col_size), nodata, dtype=np.uint32), 1)
				
				with rasterio.open(
					tile_file_2024, 'w',
					driver='GTiff',
					height=row_size,
					width=col_size,
					count=1,
					dtype=np.uint32,
					crs='EPSG:2154',
					transform=tile_transform,
					nodata=nodata,
				) as dst:
					dst.write(np.full((row_size, col_size), nodata, dtype=np.uint32), 1)
				
				tile_files_2023[(tx, ty)] = tile_file_2023
				tile_files_2024[(tx, ty)] = tile_file_2024
		
		# Process each year into temporary tiles
		def _process_one_year(geojson_path: str, tile_files: Dict[tuple[int, int], Path], desc: str) -> None:
			try:
				from tqdm import tqdm  # type: ignore
				total = count_features(geojson_path)
				pbar = tqdm(total=total if total and total > 0 else None, desc=desc)
			except Exception:
				pbar = None
			
			with open(geojson_path, "r") as f:
				for line in f:
					line = line.strip()
					if not line.startswith('{ "type": "Feature"'):
						continue
					try:
						feat = json.loads(line.rstrip(','))
					except Exception:
						if pbar: pbar.update(1)
						continue
					
					props = feat.get("properties", {})
					val = str(props.get(attr_name, ""))
					code = code_mapping.get(val, nodata)
					
					try:
						from shapely.geometry import shape
						geom = shape(feat.get("geometry")) if feat.get("geometry") else None
						if geom is None:
							if pbar: pbar.update(1)
							continue
						gxmin, gymin, gxmax, gymax = geom.bounds
					except Exception:
						if pbar: pbar.update(1)
						continue
					
					# Convert bounds to pixel rows/cols
					try:
						from rasterio.transform import rowcol
						c0, r0 = rowcol(transform, gxmin, gymax)
						c1, r1 = rowcol(transform, gxmax, gymin)
						col_start = max(0, min(c0, c1))
						col_end = min(width - 1, max(c0, c1))
						row_start = max(0, min(r0, r1))
						row_end = min(height - 1, max(r0, r1))
					except Exception:
						if pbar: pbar.update(1)
						continue
					
					# Determine overlapping tile indices
					tx_start = max(0, col_start // tile_width)
					tx_end = min(tiles_x - 1, col_end // tile_width)
					ty_start = max(0, row_start // tile_height)
					ty_end = min(tiles_y - 1, row_end // tile_height)
					
					# Process each overlapping tile
					for ty in range(ty_start, ty_end + 1):
						for tx in range(tx_start, tx_end + 1):
							tile_file = tile_files[(tx, ty)]
							
							# Read current tile content
							with rasterio.open(tile_file, 'r+') as dst:
								tile_data = dst.read(1)
								tile_transform = dst.transform
								
								# Burn feature into tile
								burn = features.rasterize(
									[(geom, code)],
									out_shape=tile_data.shape,
									transform=tile_transform,
									fill=nodata,
									dtype=np.uint32,
								)
								
								# Update tile where feature was burned
								mask = burn != nodata
								tile_data[mask] = burn[mask]
								dst.write(tile_data, 1)
					
					if pbar: pbar.update(1)
			
			if pbar: pbar.close()
		
		# Process both years
		_process_one_year(geojson_2023, tile_files_2023, desc="Rasterizing (tiled) 2023")
		_process_one_year(geojson_2024, tile_files_2024, desc="Rasterizing (tiled) 2024")
		
		# Combine tiles into final output rasters
		print("[tiled] Combining tiles into final rasters...")
		
		def _combine_tiles(tile_files: Dict[tuple[int, int], Path], output_path: str, desc: str) -> None:
			os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
			with rasterio.open(
				output_path, 'w',
				driver='GTiff',
				height=height,
				width=width,
				count=1,
				dtype=np.uint32,
				crs='EPSG:2154',
				transform=transform,
				nodata=nodata,
				compress="deflate",
				predictor=2,
			) as dst:
				for ty in range(tiles_y):
					for tx in range(tiles_x):
						tile_file = tile_files[(tx, ty)]
						col_off = tx * tile_width
						row_off = ty * tile_height
						col_size = min(tile_width, width - col_off)
						row_size = min(tile_height, height - row_off)
						
						with rasterio.open(tile_file, 'r') as src:
							tile_data = src.read(1)
							dst.write(tile_data, 1, window=rasterio.windows.Window(col_off, row_off, col_size, row_size))
		
		_combine_tiles(tile_files_2023, output_tiff_2023, "Combining 2023 tiles")
		_combine_tiles(tile_files_2024, output_tiff_2024, "Combining 2024 tiles")
		
		print(f"[tiled] Temporary files cleaned up from: {temp_dir}")
		
	finally:
		# Clean up temporary files
		if temp_dir.exists():
			shutil.rmtree(temp_dir)

	# Optionally save mapping
	if return_mapping_csv:
		print(f"[tiled] Saving mapping CSV → {return_mapping_csv}")
		pd.DataFrame({attr_name: list(code_mapping.keys()), "code": list(code_mapping.values())}) \
			.sort_values("code").to_csv(return_mapping_csv, index=False)

	print("[tiled] Done.")
	return code_mapping


__all__ = [
	"rasterize_geojson_cod_cult",
	"rasterize_two_geojsons_same_grid",
	"rasterize_geojson_cod_cult_streaming",
	"rasterize_two_geojsons_tiled",
	"compute_transition_matrix_from_rasters",
	"export_transition_matrix_csv",
]
