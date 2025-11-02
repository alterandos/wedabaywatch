import os
import numpy as np
import rasterio
from src import pipeline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from datetime import datetime
from scipy import stats
from scipy.ndimage import distance_transform_edt
from datetime import date
import glob
import pandas as pd
import sigfig

def get_class_stack(
        classified_folder_path = "output/classified/",
        classified_name_format = "*_classified.tif",
        ):
    """Opens all classification images, returns as a single stack"""
    # ---------------------------------
    # 1. Input setup
    # ---------------------------------
    files = sorted(glob.glob(os.path.join(classified_folder_path, 
                                          classified_name_format)))
    
    filenames = [os.path.basename(f) for f in files]

    # ---------------------------------
    # 2. Read and stack rasters (classifications)
    # ---------------------------------
    class_stack = []
    meta = None
    for f in files:
        with rasterio.open(f) as src:
            data = src.read(1)
            class_stack.append(data)
            if meta is None:
                meta = src.meta

    class_stack = np.stack(class_stack, axis=0)  # shape: (time, H, W)
    #H, W = stack.shape[1:]

    return class_stack, filenames

def get_index_stack(output_shape,
                    filenames,
                    index_name = "NDVI", 
                    index_folder_path = "data/derived/",
                    ):
    """ Builds a stack from index data, extracted from data/derived folder.
    Requires shape and filenames of existing stack, to match order of layers"""

    #Build empty stack
    index_stack = np.empty(output_shape,np.float32)

    #Iterate throu
    for n, fn in enumerate(filenames):
        fn_stripped = fn.split("_classified.tif")[0]
        index_file_path = index_folder_path + fn_stripped + "/" + index_name
        with rasterio.open(index_file_path) as src:
            index_data = src.read(1)
        index_stack[n, :, :] = index_data

    return index_stack
    
    

def calculate_NDVI(
        output_shape,
        img_folder_path = "data/stacked/",
        ):
    """ Opens all images in img_folder_path, calculates NDVI, returns as a stack """
    ndvi_stack = np.empty(output_shape,np.float32)
    #ndvi_stack.dtype == np.float32



    for n, img_name in enumerate(os.listdir(img_folder_path)):

        img_path = img_folder_path + img_name + "/" + img_name
        with rasterio.open(img_path) as src:
            bands, band_names, band_to_index, profile = pipeline.get_bands_from_stack(img_path)

            sensor = img_name[0:4]
            #img = src.read()
            #profile = src.profile
            #rgb = reshape_as_image(img)[:, :, :3]  # for display
            #transform = src.transform
            red = bands[band_to_index[pipeline.band_map[sensor]['R']] - 1]
            nir = bands[band_to_index[pipeline.band_map[sensor]['NIR']] - 1]

            ndvi = np.where(
                    (nir + red) == 0, np.nan,
                    (nir - red) / (nir + red)
                )
            ndvi_stack[n, :, :] = ndvi

    return ndvi_stack


def dates_convert(filenames):
    
    dates_str = [y.split("_")[3] for y in filenames]
    dates_dt = np.array([datetime.strptime(dt, '%Y%m%d') for dt in dates_str])
    dates_int = dates_dt.astype('datetime64[D]').astype(float)

    return dates_dt, dates_int


def basic_regression(x,y):
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return slope, intercept

def calculate_trend(
        class_stack,
        index_stack,
        dates_int,
        min_datapoints_for_reg = 5,
        classes_for_reg = [1]
):
    
    """Calculates linear trend of a given index in every jungle pixel.
    Requires classified stack and index stack
    Returns slope and intercept of each regression, also saved as a stack
    """ 

    H, W = class_stack.shape[1:]

    trend_stack = np.empty(class_stack.shape[1:],np.float32)
    intercept_stack = np.empty(class_stack.shape[1:],np.int16)

    for y in range(H):

        for x in range(W):
            s = class_stack[:,y,x]
            ndvi_s = index_stack[:,y,x]
            s_filter = (np.isin(s, classes_for_reg) & ~np.isnan(ndvi_s))
            if s_filter.sum() < min_datapoints_for_reg:
                trend_stack[y,x] = np.nan
                continue
            ndvi_filt = ndvi_s[s_filter]
            dates_filt = dates_int[s_filter]
            slope_xy, intercept_xy = basic_regression(dates_filt,ndvi_filt)
            try:
                intercept_stack[y,x] = intercept_xy
                trend_stack[y,x] = slope_xy
            except:
                continue


        # print progress
        if False:
            if y % 100 == 99:
                comp_perc = int(100*y/H)
                print(f"{comp_perc}% complete")

    return trend_stack, intercept_stack

        

def get_final_state(
        path = "output/classified/jungle_to_mine_change.tif",
        classes_for_reg = [1]
):
    with rasterio.open(path) as src:
        initial_state = src.read(1)
        final_state = src.read(2)
        change_year = src.read(3)

    forest_mask = (np.isin(final_state, classes_for_reg))
    final_state_no_forest = final_state.astype(float)
    final_state_no_forest[forest_mask] = np.nan 

    return final_state, final_state_no_forest

def add_scalebar(ax, length_px, label="20km", pad=-20):
    """Add a simple horizontal scale bar."""
    x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y0 = ax.get_ylim()[0] - 50
    x0 = x_center - length_px / 2
    ax.hlines(y=y0, xmin=x0, xmax=x0 + length_px, colors='black', linewidth=3)
    ax.text(x_center, y0 + pad * 0.5, label, ha='center', va='bottom', color='black', fontsize=10)
    return


def plot_index(
        trend_stack,
        final_state_no_forest,
        index_name,
        scaling = 365*13, #make it over the 13-year period,
        borders=None,
        save_path = None,
        figsize=None,
        title=None
):
    """ Plot index, with the final states overlayed."""

    trend_stack_scaled = trend_stack*scaling

    trend_median = np.nanmedian(trend_stack_scaled)

    trend_10pct = np.nanpercentile(trend_stack_scaled,10)

    # Define the boundaries (edges between color ranges)
    half_bin_width = sigfig.round(((trend_median - trend_10pct)/2),sigfigs = 1)

    bounds = [-9999, -3*half_bin_width, -half_bin_width,  half_bin_width, 3*half_bin_width, 9999]

    cmap_colors = ['#e66158', '#f4a582', "#F3E8B3", '#b8e186', '#4d9221']

    # Create colormap and normalization
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N) 

    #plt.hist(trend_stack_scaled.flatten())

    # Plot
    if title is None:
        title = index_name+" Decadal Trend Map"

    fig, ax = plt.subplots(figsize=(8, 6) if figsize is None else figsize)
    im = ax.imshow((trend_stack_scaled), cmap=cmap, norm=norm) 
    cbar = fig.colorbar(im, boundaries=bounds, ticks=bounds[1:-1])
    cbar.set_label(index_name + " Decadal Trend", fontsize=16)
    ax.set_title(title, fontsize=22, fontweight='bold', pad=17)

    pipeline.PlotUtils.add_north_arrow(ax)

    classes = {
        3: ("Cloud", "lightgrey"),
        1: ("Jungle", "forestgreen"),
        2: ("Cleared Land", "black"),
        0: ("Ocean", "skyblue"),
        #1: ("No data", "darkgrey"),
        254: ("No data", "darkgrey"),
    }

    # Sort them by code
    vals, info = zip(*sorted(classes.items()))
    names, cols = zip(*info)

    # Make colormap + matching normalizer
    class_cmap = mcolors.ListedColormap(cols)
    class_norm = mcolors.BoundaryNorm([v - 0.5 for v in vals] + [vals[-1] + 0.5], len(vals))

    # Sort class values to match colormap
    class_values = sorted(classes.keys())
    #class_colors = [classes[c][1] for c in class_values]
    remaining_class_values = np.unique(final_state_no_forest)
    class_labels = [classes[c][0] for c in remaining_class_values[remaining_class_values >= 0]]

    # Plot
    ax.imshow(final_state_no_forest, cmap=class_cmap, norm=class_norm)

    ax.set_xticks([])  # turn off x-axis
    ax.set_yticks([])  # turn off y-axis

    # Scale bar
    pixel_size = 30  # meters per pixel
    scalebar_length_m = 20000
    scalebar_length_px = scalebar_length_m / pixel_size

    # Your custom add_scalebar function (assumes it’s defined somewhere)
    add_scalebar(ax, scalebar_length_px, label = "20km")

    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path)
    return ax


def calculate_distance(
    final_state,
    plot_dist = False,
    save_plot = None
):
    """ Calculate distance from each jungle pixel (state = 1) to the nearest cleared pixel (2)"""
    # Boolean mask of "2" pixels
    mask_2 = (final_state == 2)

    # Compute Euclidean distance from every pixel to the nearest "2" pixel
    dist_to_2 = distance_transform_edt(~mask_2)

    # Mask so only class 1 pixels show distance (others as NaN)
    dist_plot = np.full_like(dist_to_2, np.nan, dtype=float)
    dist_plot[final_state == 1] = dist_to_2[final_state == 1]

    if plot_dist:
        #Plot 
        plt.figure(figsize=(8, 8))
        im = plt.imshow(dist_plot*30/1000, cmap='viridis') #*30/1000 to convert pixels -> km
        plt.colorbar(im, label='Distance (km)')
        plt.title('Distance to Nearest Mining Area')
        if save_plot is None:
            plt.show()
        else:
            plt.savefig(save_plot)

    return dist_plot

def pixel_regression(
        classified_folder_path = "output/classified/",
        classified_name_format = "*_classified.tif",
        index_folder_path = "data/derived/",
        index_name = "NDVI",
        #img_folder_path = "data/stacked/",
        min_datapoints_for_reg = 5,
        change_tif_path = "output/classified/jungle_to_mine_change.tif",
        plot_index_toggle = True,
        plot_index_path = None,
        plot_dist_toggle = False,
        index_scaling = 365*13,
        classes_for_reg = [1]
):
    class_stack, filenames = get_class_stack(classified_folder_path ,classified_name_format)

    index_stack = get_index_stack(class_stack.shape, filenames, index_name, index_folder_path)

    _, dates_int = dates_convert(filenames)

    trend_stack, intercept_stack = calculate_trend(class_stack, index_stack, dates_int, min_datapoints_for_reg, classes_for_reg)

    final_state, final_state_no_forest = get_final_state(change_tif_path, classes_for_reg=classes_for_reg)

    if plot_index_toggle:
        plot_index(trend_stack, final_state_no_forest, index_name, index_scaling, plot_index_path)
    
    
    dist_plot = calculate_distance(final_state, plot_dist_toggle, save_plot= None)
    
    return trend_stack, intercept_stack, dist_plot    
    
def plot_polygon_borders(gdf, ax, values_to_plot, color_map, linewidth=2, flip_y=True):
    """
    Plot polygon borders on an existing matplotlib axis.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing polygons with a 'value' column
    ax : matplotlib axis
        Existing axis to plot on
    values_to_plot : list
        List of values to plot (e.g., [-1] or [-1, 0, 2])
    color_map : dict
        Dictionary mapping values to colors (e.g., {-1: 'blue', 0: 'green'})
    linewidth : float, optional
        Width of the polygon borders (default: 2)
    flip_y : bool, optional
        Whether to flip y-coordinates (default: True)
    
    Returns:
    --------
    ax : matplotlib axis
        The axis with polygons plotted
    """
    # Get the y-axis limits to flip coordinates if needed
    if flip_y:
        ylim = ax.get_ylim()
        y_max = max(ylim)
    
    for value in values_to_plot:
        if value in gdf['value'].values:
            subset = gdf[gdf['value'] == value]
            # Plot directly on the provided axis
            for geom in subset.geometry:
                x, y = geom.exterior.xy
                if flip_y:
                    # Flip y coordinates
                    y = [y_max - yi for yi in y]
                ax.plot(x, y, color=color_map.get(value, 'black'), 
                       linewidth=linewidth, label=f'Value: {value}')
    
    return ax