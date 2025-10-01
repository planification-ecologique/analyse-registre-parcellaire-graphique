#!/usr/bin/env python3
"""
Load both 2023 and 2024 GPKG files, merge by ID_PARCEL, and split into:
1. CSV with matching ID_PARCEL and SURF_PARC (no geometries)
2. Separate geospatial files for 2023 and 2024 with geometries
"""

import sys
from pathlib import Path

try:
    import geopandas as gpd
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "geopandas is not installed. Install with: pip install geopandas"
    ) from exc


def prepare_data() -> None:
    data_dir = Path(__file__).parent / "data"
    gpkg_2023_path = data_dir / "PARCELLES_GRAPHIQUES_2023.gpkg"
    gpkg_2024_path = data_dir / "PARCELLES_GRAPHIQUES_2024.gpkg"
    
    # Check if both files exist
    if not gpkg_2023_path.exists():
        raise SystemExit(f"2023 GeoPackage not found: {gpkg_2023_path}")
    if not gpkg_2024_path.exists():
        raise SystemExit(f"2024 GeoPackage not found: {gpkg_2024_path}")

    print(f"Reading 2023 data: {gpkg_2023_path}")
    gdf_2023 = gpd.read_file(str(gpkg_2023_path))
    print(f"2023 - Rows: {len(gdf_2023)}, Columns: {list(gdf_2023.columns)}")
    
    print(f"Reading 2024 data: {gpkg_2024_path}")
    gdf_2024 = gpd.read_file(str(gpkg_2024_path))
    print(f"2024 - Rows: {len(gdf_2024)}, Columns: {list(gdf_2024.columns)}")
    
    # Check for required columns
    required_cols = ['ID_PARCEL', 'SURF_PARC']
    for col in required_cols:
        if col not in gdf_2023.columns:
            raise SystemExit(f"Column {col} not found in 2023 data")
        if col not in gdf_2024.columns:
            raise SystemExit(f"Column {col} not found in 2024 data")
    
    # Convert SURF_PARC to numeric
    gdf_2023['SURF_PARC'] = pd.to_numeric(gdf_2023['SURF_PARC'], errors='coerce')
    gdf_2024['SURF_PARC'] = pd.to_numeric(gdf_2024['SURF_PARC'], errors='coerce')
    
    # Merge by ID_PARCEL
    print("Merging datasets by ID_PARCEL...")
    merged = pd.merge(
        gdf_2023[['ID_PARCEL', 'SURF_PARC']].rename(columns={'SURF_PARC': 'SURF_PARC_2023'}),
        gdf_2024[['ID_PARCEL', 'SURF_PARC']].rename(columns={'SURF_PARC': 'SURF_PARC_2024'}),
        on='ID_PARCEL',
        how='inner'
    )
    
    print(f"Merged dataset - Rows: {len(merged)}")
    
    # 1. Create CSV for matching ID_PARCEL and SURF_PARC (no geometries)
    print("Creating CSV for matching parcels...")
    matching_parcels = merged[
        (merged['SURF_PARC_2023'] == merged['SURF_PARC_2024']) & 
        (merged['SURF_PARC_2023'].notna()) & 
        (merged['SURF_PARC_2024'].notna())
    ].copy()
    
    csv_path = data_dir / "matching_parcels_2023_2024.csv"
    matching_parcels.to_csv(csv_path, index=False)
    print(f"Saved matching parcels CSV: {csv_path} ({len(matching_parcels)} rows)")
    
    # 2. Create separate geospatial files for 2023 and 2024
    print("Creating geospatial files...")
    
    # Get ID_PARCEL values that did NOT make it to the matching CSV
    matching_ids = set(matching_parcels['ID_PARCEL'])
    
    # Filter geospatial data to exclude parcels that made it to the matching CSV
    gdf_2023_filtered = gdf_2023[~gdf_2023['ID_PARCEL'].isin(matching_ids)].copy()
    gdf_2024_filtered = gdf_2024[~gdf_2024['ID_PARCEL'].isin(matching_ids)].copy()
    
    # Save as Parquet (more efficient than GeoPackage for this use case)
    parquet_2023_path = data_dir / "parcels_2023_filtered.parquet"
    parquet_2024_path = data_dir / "parcels_2024_filtered.parquet"
    
    gdf_2023_filtered.to_parquet(parquet_2023_path)
    gdf_2024_filtered.to_parquet(parquet_2024_path)
    
    print(f"Saved 2023 geospatial data: {parquet_2023_path} ({len(gdf_2023_filtered)} rows)")
    print(f"Saved 2024 geospatial data: {parquet_2024_path} ({len(gdf_2024_filtered)} rows)")
    
    # Summary statistics
    print("\nSummary:")
    print(f"Total merged parcels: {len(merged)}")
    print(f"Matching SURF_PARC parcels: {len(matching_parcels)}")
    print(f"2023 filtered parcels: {len(gdf_2023_filtered)}")
    print(f"2024 filtered parcels: {len(gdf_2024_filtered)}")


if __name__ == "__main__":
    prepare_data()

