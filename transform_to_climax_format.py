#!/usr/bin/env python3
"""
Transform CRU-TS climate data to ClimaX format for climate projection training.

This script converts the NC files from cru-ts40_ar5_rcp85_mri-cgcm3_all_four_basins_150x277
to the format expected by the ClimaX climate projection codebase.
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime
import argparse

def expand_co2_to_monthly(co2_annual, time_coords):
    """
    Expand annual CO2 data to monthly resolution by repeating each annual value 12 times.
    
    Args:
        co2_annual: xarray DataArray with annual CO2 values
        time_coords: time coordinates from climate data
        
    Returns:
        Monthly CO2 values aligned with time_coords
    """
    # Extract year from time coordinates
    years = np.array([t.year for t in time_coords.values])
    
    # Create monthly CO2 array
    co2_monthly = []
    for year in years:
        # Find the closest CO2 year (in case of slight misalignment)
        co2_year_idx = np.argmin(np.abs(co2_annual.year.values - year))
        co2_monthly.append(float(co2_annual.values[co2_year_idx]))
    
    return np.array(co2_monthly)

def create_co2_spatial_field(co2_values, lat_shape, lon_shape):
    """
    Create spatially uniform CO2 field from time series.
    
    Args:
        co2_values: 1D array of CO2 values over time
        lat_shape: number of latitude points
        lon_shape: number of longitude points
        
    Returns:
        3D array (time, lat, lon) with spatially uniform CO2
    """
    # Create spatial field: CO2 is uniform across space, varies in time
    co2_field = np.broadcast_to(
        co2_values[:, np.newaxis, np.newaxis], 
        (len(co2_values), lat_shape, lon_shape)
    )
    return co2_field

def transform_to_climax_format(input_dir, output_dir):
    """
    Transform CRU-TS data to ClimaX format.
    
    Args:
        input_dir: Directory containing the original NC files
        output_dir: Directory to save transformed files
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train_val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Load data
    print("Loading data files...")
    hist_climate = xr.open_dataset(os.path.join(input_dir, 'historic-climate.nc'))
    proj_climate = xr.open_dataset(os.path.join(input_dir, 'projected-climate.nc'))
    hist_co2 = xr.open_dataset(os.path.join(input_dir, 'co2.nc'))
    proj_co2 = xr.open_dataset(os.path.join(input_dir, 'projected-co2.nc'))
    
    print(f"Historic climate: {hist_climate.time.values[0]} to {hist_climate.time.values[-1]}")
    print(f"Projected climate: {proj_climate.time.values[0]} to {proj_climate.time.values[-1]}")
    
    # Process historical data
    print("Processing historical data...")
    
    # Get spatial dimensions
    lat_size, lon_size = hist_climate.tair.shape[1], hist_climate.tair.shape[2]
    
    # Get monthly CO2 for historical period
    hist_co2_monthly = expand_co2_to_monthly(hist_co2.co2, hist_climate.time)
    hist_co2_field = create_co2_spatial_field(hist_co2_monthly, lat_size, lon_size)
    
    # Create historical inputs (CO2 only, as specified in your requirements)
    hist_inputs = xr.Dataset({
        'CO2': (['time', 'latitude', 'longitude'], hist_co2_field)
    }, coords={
        'time': hist_climate.time.values,
        'latitude': np.arange(lat_size),  # Using indices since we don't have proper lat/lon
        'longitude': np.arange(lon_size)
    })
    
    # Create historical outputs (using temperature as target)
    hist_outputs = xr.Dataset({
        'tas': (['time', 'latitude', 'longitude'], hist_climate.tair.values)
    }, coords={
        'time': hist_climate.time.values,
        'latitude': np.arange(lat_size),
        'longitude': np.arange(lon_size)
    })
    
    # Save historical data
    print("Saving historical data...")
    hist_inputs.to_netcdf(os.path.join(output_dir, 'train_val', 'inputs_historical.nc'))
    hist_outputs.to_netcdf(os.path.join(output_dir, 'train_val', 'outputs_historical.nc'))
    
    # Process projected data (treating as RCP8.5 scenario)
    print("Processing projected data...")
    
    # Get monthly CO2 for projected period  
    proj_co2_monthly = expand_co2_to_monthly(proj_co2.co2, proj_climate.time)
    proj_co2_field = create_co2_spatial_field(proj_co2_monthly, lat_size, lon_size)
    
    # Create projected inputs
    proj_inputs = xr.Dataset({
        'CO2': (['time', 'latitude', 'longitude'], proj_co2_field)
    }, coords={
        'time': proj_climate.time.values,
        'latitude': np.arange(lat_size),
        'longitude': np.arange(lon_size)
    })
    
    # Create projected outputs
    proj_outputs = xr.Dataset({
        'tas': (['time', 'latitude', 'longitude'], proj_climate.tair.values)
    }, coords={
        'time': proj_climate.time.values,
        'latitude': np.arange(lat_size),
        'longitude': np.arange(lon_size)
    })
    
    # Split projected data into train/val and test
    # Use last 20% for testing (similar to ClimateBench)
    split_idx = int(0.8 * len(proj_climate.time))
    
    # Train/val portion (first 80%)
    proj_inputs_train = proj_inputs.isel(time=slice(0, split_idx))
    proj_outputs_train = proj_outputs.isel(time=slice(0, split_idx))
    
    # Test portion (last 20%) 
    proj_inputs_test = proj_inputs.isel(time=slice(split_idx, None))
    proj_outputs_test = proj_outputs.isel(time=slice(split_idx, None))
    
    print("Saving projected data...")
    # Save as ssp585 scenario (RCP8.5 equivalent)
    proj_inputs_train.to_netcdf(os.path.join(output_dir, 'train_val', 'inputs_ssp585.nc'))
    proj_outputs_train.to_netcdf(os.path.join(output_dir, 'train_val', 'outputs_ssp585.nc'))
    
    # Save test data
    proj_inputs_test.to_netcdf(os.path.join(output_dir, 'test', 'inputs_ssp585.nc'))
    proj_outputs_test.to_netcdf(os.path.join(output_dir, 'test', 'outputs_ssp585.nc'))
    
    print("Data transformation complete!")
    print(f"Files saved to: {output_dir}")
    print(f"Historical data: {len(hist_climate.time)} time steps")
    print(f"Projected train data: {len(proj_inputs_train.time)} time steps")
    print(f"Projected test data: {len(proj_inputs_test.time)} time steps")
    
    # Print summary
    print("\nData Summary:")
    print(f"Spatial dimensions: {lat_size} x {lon_size}")
    print(f"Input variables: CO2")
    print(f"Output variable: tas (temperature)")
    print(f"Total training time steps: {len(hist_climate.time) + len(proj_inputs_train.time)}")

def main():
    parser = argparse.ArgumentParser(description='Transform CRU-TS data to ClimaX format')
    parser.add_argument('--input_dir', default='cru-ts40_ar5_rcp85_mri-cgcm3_all_four_basins_150x277',
                        help='Input directory containing NC files')
    parser.add_argument('--output_dir', default='climax_data', 
                        help='Output directory for transformed files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
        
    transform_to_climax_format(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()