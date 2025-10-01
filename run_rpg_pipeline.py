import argparse
import os

from rpg_geospatial_utils import (
	rasterize_two_geojsons_same_grid,
	compute_transition_matrix_from_rasters,
)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Rasterize RPG GeoJSON 2023/2024 by cod_cult, compute transition matrix (ha), and export CSV",
	)
	parser.add_argument("--geojson_2023", required=True, help="Path to 2023 RPG GeoJSON")
	parser.add_argument("--geojson_2024", required=True, help="Path to 2024 RPG GeoJSON")
	parser.add_argument("--out_dir", required=True, help="Output directory for rasters and CSVs")
	parser.add_argument("--resolution", type=float, default=10.0, help="Pixel size in meters (default: 10)")
	parser.add_argument("--attr", default="cod_cult", help="Attribute name for culture code (default: cod_cult)")
	parser.add_argument("--nodata", type=int, default=0, help="NODATA value for rasters (default: 0)")

	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	# Outputs
	raster_2023 = os.path.join(args.out_dir, "rpg_2023_cod_cult.tif")
	raster_2024 = os.path.join(args.out_dir, "rpg_2024_cod_cult.tif")
	mapping_csv = os.path.join(args.out_dir, f"mapping_{args.attr}.csv")
	matrix_csv = os.path.join(args.out_dir, "transition_matrix_ha_2023_2024.csv")

	# 1) Rasterize both years on the same grid and save mapping
	rasterize_two_geojsons_same_grid(
		geojson_2023=args.geojson_2023,
		geojson_2024=args.geojson_2024,
		output_tiff_2023=raster_2023,
		output_tiff_2024=raster_2024,
		resolution_m=args.resolution,
		attr_name=args.attr,
		nodata=args.nodata,
		return_mapping_csv=mapping_csv,
	)

	# 2) Compute transition matrix (hectares) and export CSV
	matrix = compute_transition_matrix_from_rasters(
		raster_2023_path=raster_2023,
		raster_2024_path=raster_2024,
		output_csv=matrix_csv,
		attr_name=args.attr,
		code_mapping_csv=mapping_csv,
	)

	# Print brief summary
	print(f"Saved rasters: {raster_2023}, {raster_2024}")
	print(f"Saved mapping: {mapping_csv}")
	print(f"Saved transition matrix (ha): {matrix_csv}")
	print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")


if __name__ == "__main__":
	main()
