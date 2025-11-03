import glob
import os
import random
from typing import Dict, Optional, Tuple

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.windows import Window
from scipy.stats import linregress
from shapely.geometry import box, shape
from sklearn.linear_model import LinearRegression

import project_config
from src import pipeline, pixel_regression

# QA bit masks (Landsat Collection 2 Level 2)
QA_BITS = {
    'cloud': 1 << 3,
    'cloud_shadow': 1 << 4,
    'cirrus': 1 << 2
}

# Classification value mapping (adjust based on your teammate's output)
LAND_COVER_CLASSES = project_config.BaseClassifier().int_class_mapping

def read_raster(class_path: str) -> Tuple[np.ndarray, rasterio.profiles.Profile]:
    """
    Read a classification raster.
    
    Parameters
    ----------
    class_path : str
        Path to classification raster (.tif)
    
    Returns
    -------
    class_array : np.ndarray
        2D array of classification values
    profile : rasterio.profiles.Profile
        Rasterio profile with CRS, transform, etc.
    """
    with rasterio.open(class_path) as src:
        class_array = src.read(1)
        profile = src.profile.copy()
    
    return class_array, profile


def get_class_mask(class_array: np.ndarray, class_value: int) -> np.ndarray:
    """
    Extract a boolean mask for a specific class.
    
    Parameters
    ----------
    class_array : np.ndarray
        Classification array
    class_value : int
        Class value to extract (e.g., 1 for mine_site_cleared)
    
    Returns
    -------
    np.ndarray
        Boolean mask where True = class_value
    """
    return class_array == class_value


def get_cleared_land_mask(class_array: np.ndarray, cleared_land_int) -> np.ndarray:
    """
    Get mask for cleared land (mine site).
    
    Parameters
    ----------
    class_array : np.ndarray
        Classification array
    
    Returns
    -------
    np.ndarray
        Boolean mask for cleared land
    """
    return get_class_mask(class_array, cleared_land_int)


def get_forest_mask(class_array: np.ndarray, fores_int) -> np.ndarray:
    """
    Get mask for uncleared forest.
    
    Parameters
    ----------
    class_array : np.ndarray
        Classification array
    
    Returns
    -------
    np.ndarray
        Boolean mask for forest
    """
    return get_class_mask(class_array, fores_int)


def classification_to_polygon(class_array: np.ndarray, 
                              transform: rasterio.Affine,
                              class_value: int,
                              crs: str = None) -> gpd.GeoDataFrame:
    """
    Convert a classification mask to polygon geometries.
    
    Parameters
    ----------
    class_array : np.ndarray
        Classification array
    transform : rasterio.Affine
        Affine transform from rasterio
    class_value : int
        Class value to vectorize
    crs : str, optional
        Coordinate reference system
    
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with polygon geometries
    """
    mask = class_array == class_value
    
    # Convert raster to polygons
    results = shapes(mask.astype(np.uint8), mask=mask, transform=transform)
    
    geometries = []
    values = []
    
    for geom, value in results:
        if value == 1:  # Only keep the polygons where mask is True
            geometries.append(shape(geom))
            values.append(class_value)
    
    gdf = gpd.GeoDataFrame({
        'class': values,
        'geometry': geometries
    }, crs=crs)
    
    return gdf


def get_mine_boundary(class_path: str, 
                      buffer_meters: float = 0,
                      simplify_tolerance: float = 10) -> gpd.GeoDataFrame:
    """
    Extract mine site boundary from classification raster.
    
    Parameters
    ----------
    class_path : str
        Path to classification raster
    buffer_meters : float
        Buffer distance in meters (positive = expand, negative = shrink)
    simplify_tolerance : float
        Tolerance for simplifying geometry (meters)
    
    Returns
    -------
    gpd.GeoDataFrame
        Mine boundary as polygon(s)
    """
    class_array, profile = read_classification_raster(class_path)
    
    # Get mine site polygons
    gdf = classification_to_polygon(
        class_array, 
        profile['transform'],
        class_value=1,  # mine_site_cleared
        crs=profile['crs']
    )
    
    # Dissolve all polygons into one
    gdf_dissolved = gdf.dissolve()
    
    # Apply buffer if requested
    if buffer_meters != 0:
        gdf_dissolved.geometry = gdf_dissolved.geometry.buffer(buffer_meters)
    
    # Simplify geometry
    if simplify_tolerance > 0:
        gdf_dissolved.geometry = gdf_dissolved.geometry.simplify(simplify_tolerance)
    
    return gdf_dissolved


def create_proximity_buffers(class_path: str,
                             buffer_distances: list,
                             output_folder: Optional[str] = None) -> Dict[int, gpd.GeoDataFrame]:
    """
    Create buffer zones around the mine site at different distances.
    
    Parameters
    ----------
    class_path : str
        Path to classification raster
    buffer_distances : list
        List of buffer distances in meters (e.g., [500, 1000, 2000])
    output_folder : str, optional
        If provided, save shapefiles to this folder
    
    Returns
    -------
    dict
        Distance -> GeoDataFrame mapping
    """
    # Get mine boundary
    mine_boundary = get_mine_boundary(class_path)
    
    buffers = {}
    
    for i, dist in enumerate(sorted(buffer_distances)):
        # Create buffer
        buffer_outer = mine_boundary.copy()
        buffer_outer.geometry = buffer_outer.geometry.buffer(dist)
        
        if i == 0:
            # First buffer: just the buffered area minus mine
            buffer_zone = buffer_outer.overlay(mine_boundary, how='difference')
        else:
            # Subsequent buffers: annulus between this and previous
            prev_dist = buffer_distances[i-1]
            buffer_inner = mine_boundary.copy()
            buffer_inner.geometry = buffer_inner.geometry.buffer(prev_dist)
            buffer_zone = buffer_outer.overlay(buffer_inner, how='difference')
        
        buffer_zone['distance_m'] = dist
        buffers[dist] = buffer_zone
        
        # Save if output folder provided
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            scene_name = os.path.splitext(os.path.basename(class_path))[0]
            out_path = os.path.join(output_folder, f'{scene_name}_buffer_{dist}m.shp')
            buffer_zone.to_file(out_path)
    
    return buffers


def mask_index_with_classification(index_array: np.ndarray,
                                   class_array: np.ndarray,
                                   keep_classes: list,
                                   cloud_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mask an index array (e.g., NDVI) using classification and cloud data.
    
    Parameters
    ----------
    index_array : np.ndarray
        Index values (e.g., NDVI)
    class_array : np.ndarray
        Classification array
    keep_classes : list
        List of class values to keep (e.g., [1, 2] for mine and forest)
    cloud_mask : np.ndarray, optional
        Boolean cloud mask (True = cloud)
    
    Returns
    -------
    np.ndarray
        Masked index array (invalid pixels = np.nan)
    """
    masked = index_array.copy().astype(float)
    
    # Mask based on classification
    valid_class_mask = np.isin(class_array, keep_classes)
    masked[~valid_class_mask] = np.nan
    
    # Mask clouds if provided
    if cloud_mask is not None:
        masked[cloud_mask] = np.nan
    
    return masked


def calculate_class_statistics(index_array: np.ndarray,
                               class_array: np.ndarray,
                               class_value: int,
                               cloud_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate statistics for an index within a specific land cover class.
    
    Parameters
    ----------
    index_array : np.ndarray
        Index values (e.g., NDVI)
    class_array : np.ndarray
        Classification array
    class_value : int
        Class to analyze
    cloud_mask : np.ndarray, optional
        Boolean cloud mask
    
    Returns
    -------
    dict
        Statistics (mean, std, min, max, count)
    """
    # Get class mask
    class_mask = class_array == class_value
    
    # Combine with cloud mask if provided
    if cloud_mask is not None:
        valid_mask = class_mask & (~cloud_mask)
    else:
        valid_mask = class_mask
    
    # Extract valid values
    valid_values = index_array[valid_mask].astype(float)
    valid_values = valid_values[~np.isnan(valid_values)]
    
    if len(valid_values) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0
        }
    
    return {
        'mean': np.mean(valid_values),
        'std': np.std(valid_values),
        'min': np.min(valid_values),
        'max': np.max(valid_values),
        'count': len(valid_values)
    }







# ==================================~ INITIAL ANALYSIS SECTION ~======================================= #

def compute_cloud_cover_all_scenes(clipped_folder, rois_folder):
    """
    Compute cloud fraction (%) for each ROI across all scenes.

    Parameters
    ----------
    clipped_folder : str
        Folder containing scene subfolders. Each subfolder has a 'qa_pixel' file.
    rois_folder : str
        Folder containing ROI shapefiles.

    Returns
    -------
    pd.DataFrame
        Rows = scene names, Columns = ROIs, Values = cloud fraction (%)
    """
    # Get scene folders
    scene_folders = [f for f in os.listdir(clipped_folder) if os.path.isdir(os.path.join(clipped_folder, f))]

    # Precompute ROI masks using the first scene as reference
    first_scene_qa = os.path.join(clipped_folder, scene_folders[0], 'qa_pixel')
    roi_masks = compute_roi_masks(rois_folder, first_scene_qa)
    roi_names = list(roi_masks.keys())

    # Initialize results dict
    cloud_data = {}

    for scene_name in scene_folders:
        qa_path = os.path.join(clipped_folder, scene_name, 'qa_pixel')
        cloud_mask = get_cloud_mask(qa_path)  # True = cloud pixels

        # Compute cloud fraction per ROI
        scene_cloud_fractions = {}
        for roi_name, roi_mask in roi_masks.items():
            combined_mask = roi_mask  # ROI mask
            n_cloud = np.sum(cloud_mask & combined_mask)
            n_total = np.sum(combined_mask)
            fraction = (n_cloud / n_total * 100) if n_total > 0 else np.nan
            scene_cloud_fractions[roi_name] = fraction

        cloud_data[scene_name] = scene_cloud_fractions

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(cloud_data, orient='index')
    return df


def get_roi_shapefiles(rois_folder):
    """
    Returns a dict mapping ROI names (from .shp filenames) to paths.
    """
    roi_shp_paths = {}
    for shp_file in glob.glob(os.path.join(rois_folder, '*.shp')):
        roi_name = os.path.splitext(os.path.basename(shp_file))[0]
        roi_shp_paths[roi_name] = shp_file
    return roi_shp_paths

def get_cloud_mask_from_qa_pixel_explicit(qa_path):
    QA_BITS = {
        'cloud': 1 << 3,
        'cloud_shadow': 1 << 4,
        'cirrus': 1 << 2
    }

    ignore_value = 0
    
    with rasterio.open(qa_path) as src:
        qa = src.read(1) # [0, 0, 1, 0]
        nodata = src.nodata

    cloud_mask = np.full_like(qa, fill_value=ignore_value, dtype=int)
    cloud_mask[(qa & QA_BITS['cloud']) > 0] = 1
    cloud_mask[(qa & QA_BITS['cloud_shadow']) > 0] = 2
    cloud_mask[(qa & QA_BITS['cirrus']) > 0] = 3

    if nodata is not None:
        cloud_mask[qa == nodata] = -1

    cloud_map = {1: 'cloud', 2: 'cloud_shadow', 3: 'cirrus'}

    return cloud_mask, cloud_map, ignore_value


def get_cloud_mask(qa_path, cloud=True, cloud_shadow=True, cirrus=False):
    """
    Returns a boolean mask of cloud pixels from a QA_PIXEL raster.
    
    Parameters
    ----------
    qa_path : str
        Path to QA_PIXEL raster.
    
    Returns
    -------
    np.ndarray
        Boolean array where True indicates a cloud, cloud shadow, or cirrus pixel.
    """
    with rasterio.open(qa_path) as src:
        qa = src.read(1) # [0, 0, 1, 0]
        nodata = src.nodata

        cloud_mask = np.zeros_like(qa, dtype=bool)
        
        if cloud:
            cloud_mask |= (qa & QA_BITS['cloud']) > 0
        if cloud_shadow:
            cloud_mask |= (qa & QA_BITS['cloud_shadow']) > 0
        if cirrus:
            cloud_mask |= (qa & QA_BITS['cirrus']) > 0
        
        if nodata is not None:
            cloud_mask[qa == nodata] = True
    
    return cloud_mask

def mask_clouds_in_scene_raster(raster_path, qa_path):
    """
    Mask clouds and nodata for a single raster of a scene using QA_PIXEL.

    Parameters
    ----------
    raster_path : str
        Path to the raster to mask (e.g., NDVI, NDBI, etc.).
    qa_path : str
        Path to the QA_PIXEL raster.

    Returns
    -------
    np.ndarray
        Masked array (clouds and nodata are set to np.nan) with same shape as input raster.
    """
    # Get cloud mask
    cloud_mask = get_cloud_mask(qa_path)

    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

        # Mask clouds
        arr[cloud_mask] = np.nan

        # Mask nodata if set
        if nodata is not None:
            arr[arr == nodata] = np.nan

    return arr

def compute_roi_mask_in_scene(roi_shapefile, reference_scene_path):
    """
    Compute a boolean mask for a single ROI shapefile on the grid of a reference scene.

    Parameters
    ----------
    roi_shapefile : str
        Path to the ROI shapefile.
    reference_scene_path : str
        Path to the reference raster scene.

    Returns
    -------
    np.ndarray
        Boolean array (True = inside ROI)
    """
    with rasterio.open(reference_scene_path) as src:
        shape = (src.height, src.width)
        transform = src.transform

    gdf = gpd.read_file(roi_shapefile)
    mask = np.zeros(shape, dtype=bool)

    for geom in gdf.geometry:
        mask |= rasterio.features.geometry_mask([geom], transform=transform, invert=True, out_shape=shape)

    return mask

def compute_roi_masks(rois_folder, reference_scene_path):
    """
    Compute boolean masks for all ROI shapefiles in a folder on the grid of a reference scene.

    Parameters
    ----------
    rois_folder : str
        Folder containing ROI shapefiles.
    reference_scene_path : str
        Path to the reference raster scene.

    Returns
    -------
    dict
        Mapping: ROI name -> boolean mask (True = inside ROI)
    """
    roi_paths = {os.path.splitext(os.path.basename(f))[0]: f
                 for f in glob.glob(os.path.join(rois_folder, '*.shp'))}
    
    roi_masks = {}
    for roi_name, shp_path in roi_paths.items():
        roi_masks[roi_name] = compute_roi_mask_in_scene(shp_path, reference_scene_path)

    return roi_masks

def compute_cloud_fraction_per_roi(roi_masks, cloud_mask):
    """
    Compute the fraction of pixels that are cloudy or nodata in each ROI.

    Parameters
    ----------
    roi_masks : dict
        Mapping: ROI name -> boolean mask (True = inside ROI)
    cloud_mask : np.ndarray
        Boolean array of the same shape as the scene (True = cloud/nodata)

    Returns
    -------
    cloud_fraction : dict
        ROI name -> fraction of pixels that are cloudy or nodata (0-1)
    """
    cloud_fraction = {}
    for roi_name, roi_mask in roi_masks.items():
        n_total = roi_mask.sum()
        n_cloudy = (roi_mask & cloud_mask).sum()
        cloud_fraction[roi_name] = n_cloudy / n_total if n_total > 0 else np.nan

    return cloud_fraction

def compute_index_stats_for_roi_cloud_masked(index_path, combined_masks):
    """
    Compute per-ROI statistics for a single index raster given precomputed ROI+cloud masks.

    Parameters
    ----------
    index_path : str
        Path to a single index raster (ENVI or GeoTIFF).
    combined_masks : dict
        ROI name -> boolean array (True = usable pixel: inside ROI AND not cloudy).

    Returns
    -------
    dict
        ROI name -> dict of statistics (mean, std, min, max, count)
    """
    stats = {}

    with rasterio.open(index_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

    for roi_name, mask in combined_masks.items():
        roi_data = np.where(mask, data, np.nan)
        stats[roi_name] = {
            'mean': np.nanmean(roi_data),
            'std': np.nanstd(roi_data),
            'min': np.nanmin(roi_data),
            'max': np.nanmax(roi_data),
            'count': np.count_nonzero(~np.isnan(roi_data))
        }

    return stats

def compute_scene_roi_stats(derived_folder, rois_folder, clipped_folder, composites):
    """
    Compute ROI statistics for all scenes and indices in a derived folder, masking clouds.

    Parameters
    ----------
    derived_folder : str
        Folder containing scene subfolders (one per date/scene).
    rois_folder : str
        Folder containing ROI shapefiles.
    clipped_folder : str
        Folder containing QA_PIXEL files for cloud masking (same subfolder names as scenes).

    Returns
    -------
    pd.DataFrame
        Rows = scenes, columns = ROIs. Each cell = dict of statistics including cloud_fraction.
    """
    all_stats = {}

    # Get all scene folders
    scene_folders = [f for f in os.listdir(derived_folder) if os.path.isdir(os.path.join(derived_folder, f))]

    for scene_name in scene_folders:
        scene_path = os.path.join(derived_folder, scene_name)
        qa_path = os.path.join(clipped_folder, scene_name, 'qa_pixel')

        # 1. Precompute ROI masks on this scene
        roi_masks = compute_roi_masks(rois_folder, qa_path)

        # 2. Get cloud mask
        cloud_mask = get_cloud_mask(qa_path)  # True = cloud pixels

        # 3. Combine ROI masks with cloud mask (positive mask = not cloud)
        combined_masks = {roi: roi_mask & (~cloud_mask) for roi, roi_mask in roi_masks.items()}

        # 4. Compute cloud fraction per ROI
        cloud_fractions = compute_cloud_fraction_per_roi(roi_masks, cloud_mask)

        # 5. Compute stats for each index in this scene
        scene_stats = {}

        for roi_name, mask in combined_masks.items():
            roi_stats = {'cloud_fraction': cloud_fractions[roi_name]}
            for composite_name in composites:
                index_path = os.path.join(scene_path, composite_name)  # no extension
                stats = compute_index_stats_for_roi_cloud_masked(index_path, {roi_name: mask})
                roi_stats[composite_name] = stats[roi_name]
            scene_stats[roi_name] = roi_stats

        all_stats[scene_name] = scene_stats

    # Convert to dataframe
    df = pd.DataFrame.from_dict(all_stats, orient='index')
    return df

def filter_all_stats_on_threshold_value(df, threshold_value, threshold_statistic):
    """
    Returns a new DataFrame keeping only ROIs in each scene where cloud_fraction <= max_cloud_fraction.
    ROIs exceeding the threshold will be set to NaN (or removed).
    """
    df_filtered = df.copy()
    
    for scene in df.index:
        for roi in df.columns:
            stats_dict = df.at[scene, roi]
            if stats_dict[threshold_statistic] > threshold_value:
                df_filtered.at[scene, roi] = None  # or np.nan

    return df_filtered

def extract_stat_df(df, composite, stat):
    """
    Extract a DataFrame of a specific composite and stat across all scenes and ROIs,
    safely handling None values and case-insensitive keys.
    """
    composite_lower = composite.lower()
    stat_lower = stat.lower()

    def get_stat(cell):
        if not isinstance(cell, dict):
            return np.nan
        # Find composite key (case-insensitive)
        comp_key = next((k for k in cell.keys() if k.lower() == composite_lower), None)
        if comp_key is None or not isinstance(cell[comp_key], dict):
            return np.nan
        # Find stat key (case-insensitive)
        stat_key = next((k for k in cell[comp_key].keys() if k.lower() == stat_lower), None)
        if stat_key is None:
            return np.nan
        return cell[comp_key][stat_key]

    return df.map(get_stat)

def plot_composite_over_time(df, composite, stat, rois=None):
    """
    Plot the change of a given composite/stat over time for selected or all ROIs.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-index dataframe (scenes × ROIs) with stats per composite.
    composite : str
        Name of composite (e.g. 'NDVI', 'EVI').
    stat : str
        Statistic to plot (e.g. 'mean', 'max').
    rois : list, optional
        Subset of ROIs to include (case-insensitive, default = all).
    """
    composite = composite.upper()
    stat = stat.lower()

    # Extract sub-DF
    stat_df = extract_stat_df(df, composite, stat)

    # Filter ROIs if provided
    if rois is not None:
        rois = [r.lower() for r in rois]
        stat_df = stat_df[[c for c in stat_df.columns if c.lower() in rois]]

    plt.figure(figsize=(12, 8))

    for roi in stat_df.columns:
        y = stat_df[roi].replace({None: np.nan}).astype(float)
        y_line = y.interpolate(method='linear', limit_direction='both')
        x = stat_df.index

        plt.plot(x, y_line, label=roi.replace('_', ' ').upper())
        plt.scatter(x[y.notna()], y[y.notna()], marker='o', s=100)

    plt.xlabel('Scene')
    plt.ylabel(f'{composite} {stat.capitalize()}')
    plt.title(f'{composite} {stat.capitalize()} Over Time per ROI')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return stat_df

def plot_time_series_with_trend(time_index, y_values, trend_values, trend_stats=None, title='', ylabel=''):
    """
    Plot a time series with a linear trend line.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Datetimes corresponding to the observations.
    y_values : np.ndarray or pd.Series
        Observed values.
    trend_values : np.ndarray
        Fitted trend values at the same time points as y_values.
    roi_name : str
        Name of the ROI for labeling.
    slope : float, optional
        Slope of the trend line.
    p_value : float, optional
        P-value of the trend for annotation.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(time_index, y_values, 'o', label='Data')
    label = 'Trend:\n    '
    if trend_stats is not None:
        label += f'{'\n    '.join([f'{k}: {v}' for k, v in trend_stats.items()])}'
    plt.plot(time_index, trend_values, '-', color='red', label=label)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()


def calculate_linear_trend(df, composite, roi, plot=False):
    """
    Compute a linear trend (and significance) for a time series with irregular intervals.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    y_series = df[roi].replace([np.inf, -np.inf], np.nan).dropna()
    x = y_series.index.astype(np.int64) / 1e9  # seconds since epoch
    y = y_series.values

    # Perform regression with p-value
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    if plot:
        slope_per_year = slope * 31_556_952
        trend_stats = {'slope': f'{slope_per_year:.3f}/yr', 'p': f'{p_value:.4f}', 'R²': f'{r_value**2:.2f}'}
        trend_values = intercept + slope * x
        title=f'Linear Trend: {composite} in {roi.replace('_', ' ').title()}'
        plot_time_series_with_trend(y_series.index, y, trend_values, trend_stats=trend_stats, title=title, ylabel=composite)

    return slope, intercept, r_value**2, p_value


def plot_roi_timeseries(df, rois, roi_colours, roi_comparisons=None, ylabel='', title='', comparison_alpha=0.1, figsize=None):
    """
    Plot NDVI (or other index) over time for a subset of ROIs.

    Parameters
    ----------
    df : pd.DataFrame
        Index = YYYYMMDD (int or str), columns = ROI names
    rois : list
        List of ROI names to plot
    roi_colours : dict
        ROI name -> hex color
    ylabel : str
        Y-axis label
    """
    # Convert index to datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
    df = df[df.index.notna()]

    fig, ax = plt.subplots(figsize=(16, 10) if figsize is None else figsize)

    for roi in rois:
        if roi not in df.columns:
            print(f'Skipping {roi} (not found in DataFrame)')
            continue

        y = df[roi].replace([None, np.inf, -np.inf], np.nan).astype(float)
        y_line = y.interpolate(method='linear', limit_direction='both')

        colour = roi_colours.get(roi, 'black')

        # Plot line (continuous through NaNs)
        ax.plot(df.index, y_line, linestyle='-', color=colour, label=roi.replace('_', ' ').title())

        # Plot markers only where actual data exists
        ax.scatter(df.index[y.notna()], y[y.notna()], marker='o', color=colour, s=69)

    if roi_comparisons:
        for roi in roi_comparisons:
            if roi not in df.columns:
                print(f'Skipping {roi} (not found in DataFrame)')
                continue

            y = df[roi].replace([None, np.inf, -np.inf], np.nan).astype(float)
            y_line = y.interpolate(method='linear', limit_direction='both')

            colour = roi_colours.get(roi, 'black')

            # Plot line (continuous through NaNs)
            ax.plot(df.index, y_line, linestyle='-', color=colour, label=roi.replace('_', ' ').title(), alpha=comparison_alpha)

            # Plot markers only where actual data exists
            ax.scatter(df.index[y.notna()], y[y.notna()], marker='o', color=colour, s=69, alpha=comparison_alpha)

    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.legend(title='ROI', bbox_to_anchor=(1, 0.5), loc='center left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_rois_on_basemap(rois_folder, rgb_path, roi_colours=None, title=None, highlight_rois=None, alphas=None):
    """
    Plot all ROI polygons on an RGB basemap with predefined colours first,
    then random colours once the predefined ones are exhausted.
    """

    if alphas is None:
        alphas = {'default': 0.15, 'highlight': 0.95}

    with rasterio.open(rgb_path) as src:
        r, g, b = src.read(1), src.read(2), src.read(3)
        rgb = np.dstack([r, g, b]).astype(np.float32)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        crs = src.crs

    rgb_min, rgb_max = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - rgb_min) / (rgb_max - rgb_min), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, extent=extent)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=24, fontweight='bold', pad=17)
    else:
        ax.set_title('RGB Scene with ROIs', fontsize=24)

    roi_paths = get_roi_shapefiles(rois_folder)
    legend_handles = []

    for roi_name, path in roi_paths.items():
        gdf = gpd.read_file(path).to_crs(crs)

        color = roi_colours.get(
            roi_name,
            '#' + ''.join(random.choices('0123456789ABCDEF', k=6))
        )

        if highlight_rois and roi_name in highlight_rois:
            alpha_val = alphas['highlight']
        else:
            alpha_val = alphas['default']

        gdf.plot(ax=ax, facecolor=color, edgecolor='black', alpha=alpha_val)

        # Add to legend
        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=6, label=roi_name.replace('_', ' ').title(), alpha=alpha_val)
        )

        # Label ROI
        for _, row in gdf.iterrows():
            centroid = row.geometry.centroid
            ax.text(
                centroid.x, centroid.y, roi_name.replace('_', ' ').title(),
                fontsize=16, ha='center', va='center', color='white', alpha=alpha_val,
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')]
            )

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Add white north arrow (bottom-right)
    pipeline.PlotUtils.add_north_arrow(ax, color='white')
    pipeline.PlotUtils.add_scalebar(ax, colour='white')
    pipeline.PlotUtils.add_geographic_labels(ax, rgb_path)

    # Legend outside on right
    ax.legend(
        handles=legend_handles,
        title='Regions of Interest',
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()
    plt.show()

def plot_stretched_index(index_array, title, ax, cmap='Greys', percentile=(2, 98)):
    """Plot index with percentile stretching."""
    valid_data = index_array[~np.isnan(index_array)]
    
    if len(valid_data) == 0:
        ax.imshow(index_array, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
        return
    
    vmin, vmax = np.percentile(valid_data, percentile)
    ax.imshow(index_array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')


def create_cloud_filled_composite(scene_list, derived_folder, clipped_folder, classified_state, classified_state_mask_values=[-1], index_name='BSI', mask_states=False):
    """
    Create composite by filling clouds with most recent valid pixel.
    classified_state_water_value is a list containing the values you want to mask. Default is [-1]
    """
    scenes_data = []
    
    for scene in scene_list:
        index_path = os.path.join(derived_folder, scene, index_name)
        qa_path = os.path.join(clipped_folder, scene, 'QA_PIXEL')
        
        with rasterio.open(index_path) as src:
            index_data = src.read(1).astype(float)

        # print(f' == last mean is {np.nanmean(index_data)}')
        
        cloud_mask, _, _ = get_cloud_mask_from_qa_pixel_explicit(qa_path)
        if mask_states:
            mask = np.isin(classified_state, classified_state_mask_values)
            index_data[mask] = np.nan

        is_cloud = np.isin(cloud_mask, [1, 2])
        index_data[is_cloud] = np.nan
        # print(f' == masked mean is {np.nanmean(index_data)}')
        
        scenes_data.append(index_data)
    
    stack = np.stack(scenes_data, axis=0)
    composite = np.full(stack.shape[1:], np.nan)
    
    for i in range(stack.shape[1]):
        for j in range(stack.shape[2]):
            pixel_timeseries = stack[:, i, j]
            valid_mask = ~np.isnan(pixel_timeseries)
            if valid_mask.any():
                composite[i, j] = pixel_timeseries[valid_mask][0]
    
    return composite


def plot_first_last_comparison(last, first, derived_folder, clipped_folder, classified_state, classified_state_mask_values=[-1], index_name='BSI', mask_states=False, cmap='RdYlBu_r', percentile=(2, 98)):
    """
    Plot first vs last comparison for a given index.
    
    Parameters
    ----------
    last_sorted : list
        The last scene images
    index_name : str
        Index to plot (e.g., 'BSI', 'NDVI', 'EVI')
    cmap : str
        Matplotlib colormap (e.g., 'RdYlBu_r', 'RdYlGn', 'Greys')
    percentile : tuple
        (min, max) percentiles for stretching (default: (2, 98))
    """
    
    # Create cloud-filled composite for first
    print(f"Creating cloud-filled composite for first {index_name}...")
    first_index_composite = create_cloud_filled_composite(first, derived_folder, clipped_folder, classified_state, classified_state_mask_values, index_name=index_name, mask_states=mask_states)

    print(f"Creating cloud-filled composite for last {index_name}...")
    last_index_composite = create_cloud_filled_composite(last, derived_folder, clipped_folder, classified_state, classified_state_mask_values, index_name=index_name, mask_states=mask_states)
    
    first_date = pipeline.extract_scene_date(first[0])
    last_date = pipeline.extract_scene_date(last[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Compute shared vmin/vmax
    valid_first = first_index_composite[~np.isnan(first_index_composite)]
    valid_last = last_index_composite[~np.isnan(last_index_composite)]
    combined = np.concatenate([valid_first, valid_last])
    vmin, vmax = np.percentile(combined, percentile)
    

    # Plot both using the same limits
    im1 = axes[0].imshow(first_index_composite, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'a) {index_name} - First Scene\n{first_date}', fontsize=16, fontweight='bold', pad=17)
    axes[0].axis('off')

    pipeline.PlotUtils.add_north_arrow(axes[0])
    # pipeline.PlotUtils.add_scalebar(axes[0])
    pixel_size = 30  # meters per pixel
    scalebar_length_m = 20000
    scalebar_length_px = scalebar_length_m / pixel_size
    pixel_regression.add_scalebar(axes[0], scalebar_length_px, label = "20km")

    im2 = axes[1].imshow(last_index_composite, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'b) {index_name} - Last Scene\n{last_date}', fontsize=16, fontweight='bold', pad=17)
    axes[1].axis('off')
    pipeline.PlotUtils.add_north_arrow(axes[1])
    # pipeline.PlotUtils.add_scalebar(axes[1])
    pixel_regression.add_scalebar(axes[1], scalebar_length_px, label = "20km")


    # Shared colorbar
    cax = fig.add_axes([0.15, -0.03, 0.7, 0.03])   # <-- reduce the 2nd value to move lower
    cbar = fig.colorbar(im2, cax=cax, orientation='horizontal')
    cbar.set_label(f'{index_name} value')

    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n" + "="*60)
    print(f"{index_name} STATISTICS")
    print("="*60)
    print(f"\nFirst scene ({first_date}):")
    print(f"  Mean: {np.nanmean(first_index_composite):.3f}")
    print(f"  Std:  {np.nanstd(first_index_composite):.3f}")
    print(f"  Valid pixels: {np.sum(~np.isnan(first_index_composite)):,}")
    
    print(f"\nLast composite ({last_date}):")
    print(f"  Mean: {np.nanmean(last_index_composite):.3f}")
    print(f"  Std:  {np.nanstd(last_index_composite):.3f}")
    print(f"  Valid pixels: {np.sum(~np.isnan(last_index_composite)):,}")
    print(f"  Remaining NaN: {np.sum(np.isnan(last_index_composite)):,}")