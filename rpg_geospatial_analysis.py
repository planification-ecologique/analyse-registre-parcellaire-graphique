#!/usr/bin/env python3
"""
Load both 2023 and 2024 GPKG files, merge by ID_PARCEL, and split into:
1. CSV with matching ID_PARCEL and SURF_PARC (no geometries)
2. Separate geospatial files for 2023 and 2024 with geometries
"""

import sys
from pathlib import Path
import time
try:
    import geopandas as gpd
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "geopandas is not installed. Install with: pip install geopandas"
    ) from exc

data_dir = Path(__file__).parent / "data"

def prepare_data() -> None:
    
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
    
    # Merge by ID_PARCEL - include all properties from both datasets
    print("Merging datasets by ID_PARCEL...")
    
    # Prepare 2023 data with year suffix
    gdf_2023_renamed = gdf_2023.copy()
    gdf_2023_renamed = gdf_2023_renamed.drop(columns=['geometry'])
    gdf_2023_renamed.columns = [f"{col}_2023" if col != 'ID_PARCEL' else col for col in gdf_2023_renamed.columns]

    # Prepare 2024 data with year suffix  
    gdf_2024_renamed = gdf_2024.copy()
    gdf_2024_renamed = gdf_2024_renamed.drop(columns=['geometry'])
    gdf_2024_renamed.columns = [f"{col}_2024" if col != 'ID_PARCEL' else col for col in gdf_2024_renamed.columns]
    
    # Merge all properties
    merged = pd.merge(
        gdf_2023_renamed,
        gdf_2024_renamed,
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


def load_initial_data() -> None:
    print(f"Loading data from {data_dir}")
    matching_parcels = pd.read_csv(data_dir / "matching_parcels_2023_2024.csv")
    gdf_2023_filtered = gpd.read_parquet(data_dir / "parcels_2023_filtered.parquet")
    gdf_2024_filtered = gpd.read_parquet(data_dir / "parcels_2024_filtered.parquet")

    print(f"Loaded {len(matching_parcels)} matching parcels")
    print(f"Loaded {len(gdf_2023_filtered)} 2023 filtered parcels")
    print(f"Loaded {len(gdf_2024_filtered)} 2024 filtered parcels")
    return matching_parcels, gdf_2023_filtered, gdf_2024_filtered

def analyse_geospatial_data(gdf_2023_filtered: gpd.GeoDataFrame, gdf_2024_filtered: gpd.GeoDataFrame) -> None:
    # Use geopandas overlay to find intersections
    print("Calculating intersections using geopandas.overlay...")
    start_time = time.time()
    intersections = gpd.overlay(gdf_2023_filtered, gdf_2024_filtered, how='intersection')
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Found {len(intersections)} intersection polygons")
    
    # Rename columns to have meaningful suffixes
    column_mapping = {}
    for col in intersections.columns:
        if col.endswith('_1'):
            column_mapping[col] = col.replace('_1', '_2023')
        elif col.endswith('_2'):
            column_mapping[col] = col.replace('_2', '_2024')
    intersections = intersections.rename(columns=column_mapping)
    print(f"Column names: {list(intersections.columns)}")
    
    # Calculate intersection areas in hectares
    intersections['intersection_area'] = intersections.geometry.area/10000
    print(f"Intersection areas calculated. Sample: {intersections[['ID_PARCEL_2023', 'intersection_area']].head()}")
    print(f"Total intersection area: {intersections['intersection_area'].sum():.2f} ha")

    # Save intersection data to parquet
    intersections.to_parquet(data_dir / "intersection_parcels.parquet")
    print(f"Saved intersection data: {data_dir / 'intersection_parcels.parquet'}")
    
    # drop geometry column and print head
    intersections = intersections.drop(columns=['geometry'])
    print(f"Intersection data: {intersections.head()}")
    print(f"Loaded {len(intersections)} intersection polygons")
    # Save as CSV
    intersections.to_csv(data_dir / "intersection_parcels.csv", index=False)
    print(f"Saved intersection data: {data_dir / 'intersection_parcels.csv'}")
    return intersections

def combine_data(matching_parcels: pd.DataFrame, intersected_parcels: pd.DataFrame) -> None:
    """Reformat and append the two dataframes keeping all columns and renaming area columns"""
    
    # Reformat matching_parcels - keep all columns, rename SURF_PARC_2023 to AREA
    matching_formatted = matching_parcels.copy()
    matching_formatted['ID_PARCEL_2023'] = matching_formatted['ID_PARCEL']
    matching_formatted['ID_PARCEL_2024'] = matching_formatted['ID_PARCEL']
    matching_formatted = matching_formatted.rename(columns={'SURF_PARC_2023': 'AREA'})
    matching_formatted['TYPE'] = 'MATCHING'
    
    # Reformat intersected_parcels - keep all columns, rename intersection_area to AREA
    intersected_formatted = intersected_parcels.copy()
    intersected_formatted = intersected_formatted.rename(columns={'intersection_area': 'AREA'})
    intersected_formatted['TYPE'] = 'INTERSECTED'
    
    # Combine the two dataframes
    combined = pd.concat([matching_formatted, intersected_formatted], ignore_index=True)
    
    print(f"Combined data: {combined.head()}")
    print(f"Loaded {len(combined)} combined parcels")
    print(f"Matching parcels: {len(matching_formatted)}")
    print(f"Intersected parcels: {len(intersected_formatted)}")
    
    # Save as CSV
    combined.to_csv(data_dir / "combined_parcels.csv", index=False)
    print(f"Saved combined data: {data_dir / 'combined_parcels.csv'}")

def calculate_transition_matrix(combined_parcels: pd.DataFrame) -> None:
    """Calculate the transition matrix by culture"""
    # First aggregate areas by CODE_CULTU combinations to handle duplicates
    aggregated = combined_parcels.groupby(['CODE_CULTU_2023', 'CODE_CULTU_2024'])['AREA'].sum().reset_index()
    print(f"Aggregated data: {len(aggregated)} unique culture transitions")
    
    # Now create the pivot table
    transition_matrix = aggregated.pivot(index='CODE_CULTU_2023', columns='CODE_CULTU_2024', values='AREA')
    
    # Fill NaN values with 0 (no transition)
    transition_matrix = transition_matrix.fillna(0)
    # Only keep 2 digits after the decimal point
    transition_matrix = transition_matrix.round(2)
    
    print(f"Transition matrix shape: {transition_matrix.shape}")
    print(f"Transition matrix:\n{transition_matrix}")
    
    # Save as CSV
    transition_matrix.to_csv(data_dir / "transition_matrix.csv")
    print(f"Saved transition matrix: {data_dir / 'transition_matrix.csv'}")
    return transition_matrix

if __name__ == "__main__":
    # prepare_data()
    # matching_parcels, gdf_2023_filtered, gdf_2024_filtered = load_initial_data()
    # intersected_parcels = analyse_geospatial_data(gdf_2023_filtered, gdf_2024_filtered)
    
    # matching_parcels = pd.read_csv(data_dir / "matching_parcels_2023_2024.csv")
    # intersected_parcels = pd.read_csv(data_dir / "intersection_parcels.csv")
    
    # combine_data(matching_parcels, intersected_parcels)
    combined_parcels = pd.read_csv(data_dir / "combined_parcels.csv")
    transition_matrix = calculate_transition_matrix(combined_parcels)
