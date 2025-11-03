import rasterio
import pyproj
from rasterio.mask import mask
from shapely.geometry import box, mapping, Polygon
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import numpy as np

import os
from pathlib import Path
import tarfile
import re

# --- Raster and analytical functions --- #
def calculate_area_projected(coords):
    """Calculate area in km² for a polygon in projected CRS (meters)."""
    poly = Polygon(coords)
    area_m2 = poly.area
    return area_m2 / 1e6

def get_bbox_and_crs_from_raster(raster_path):
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        return bounds, crs

def clip_array_to_bbox(raster, bbox):
    """
    Clip an open rasterio dataset or raster array to a bounding box.

    Parameters:
        raster: open rasterio dataset or object compatible with rasterio.mask.mask
        bbox (tuple): (minx, miny, maxx, maxy)

    Returns:
        clipped_array (np.ndarray): clipped raster data
        clipped_transform (Affine): transform for clipped raster
    """
    geom = [mapping(box(*bbox))]
    clipped_array, clipped_transform = mask(raster, geom, crop=True)
    return clipped_array, clipped_transform

def get_matching_band_token(band_name, bands_to_keep):
    """
    Given a band filename and list of desired band tokens (e.g. ['B2','B3','B4','B5']),
    return the matching token or None if no match.
    """
    band_basename = os.path.splitext(os.path.basename(band_name))[0]

    if not any(b in band_basename for b in bands_to_keep):
        return None

    for b in bands_to_keep:
        if band_basename.upper().endswith(b.upper()) or band_basename.upper().endswith(f'{b.upper()}_'):
            return b
    return None

def clip_scene_and_save_to_output(tar_ref, band_tif, out_path, bbox):
    with tar_ref.extractfile(band_tif) as tif_file:
        with rasterio.open(tif_file) as src:
            clipped, transform = clip_array_to_bbox(src, bbox)
            profile = src.profile
            profile.update({
                'driver': 'ENVI',
                'height': clipped.shape[1],
                'width': clipped.shape[2],
                'transform': transform
            })

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(clipped)

def clip_raw_scenes_to_study_area(raw_folder, clipped_folder, study_area_bbox, bands_to_keep, replace_existing=False):
    counter = 0

    tar_paths = get_all_files_in_dir_by_extension(raw_folder, 'tar')

    for tar_path in tar_paths:
        tar_base = os.path.splitext(os.path.basename(tar_path))[0] # tar file name without extension
        out_dir = make_new_dir(clipped_folder, tar_base) # output folder for the clipped bands

        sensor = extract_landsat_token(tar_base) # which landsat it is so we get the right bands
        sensor_bands_to_keep = bands_to_keep[sensor] # the right bands to keep

        with tarfile.open(tar_path, 'r') as tar_ref:
            tifs_in_tar = [m for m in tar_ref.getmembers() if m.isfile() and m.name.lower().endswith(('.tif', '.tiff'))]

            for band_tif in tifs_in_tar:
                band_name = os.path.basename(band_tif.name)
                band_token = get_matching_band_token(band_name, sensor_bands_to_keep)
                if band_token is None:
                    continue

                out_path = os.path.join(out_dir, band_token)
                counter += process_if_needed(
                            out_path,
                            _replace=replace_existing,
                            process_func=clip_scene_and_save_to_output,
                            tar_ref=tar_ref,
                            band_tif=band_tif,
                            out_path=out_path,
                            bbox=study_area_bbox
                        )
    return counter

def stack_rasters(band_paths, output_path):
    """
    Stack single-band rasters into one multiband raster.
    Assumes all rasters share the same shape, CRS, and transform.
    """
    with rasterio.open(band_paths[0]) as src0:
        meta = src0.meta.copy()
        meta.update(count=len(band_paths))

    band_names = [os.path.splitext(os.path.basename(p))[0] for p in band_paths]

    with rasterio.open(output_path, 'w', **meta) as dst:
        for idx, path in enumerate(band_paths, start=1):
            with rasterio.open(path) as src:
                dst.write(src.read(1), idx)
        for i, b in enumerate(band_names, start=1):
            dst.set_band_description(i, b)

def stack_all_bands_in_dir(clipped_folder, output_folder, bands_to_keep, replace_existing=False):
    """
    Stack clipped single-band rasters for each scene into multiband stacks.

    Parameters:
        clipped_folder (str): Folder containing subfolders of clipped scenes.
        output_folder (str): Folder to save stacked rasters.
        bands_to_keep (dict): Dict mapping sensor codes (e.g. 'LC08') to band lists.
    """
    counter = 0
    scene_dirs = [os.path.join(clipped_folder, d) for d in os.listdir(clipped_folder) if os.path.isdir(os.path.join(clipped_folder, d))]

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        sensor = extract_landsat_token(scene_name)
        bands = bands_to_keep.get(sensor, [])

        # Ensure all required bands exist
        # band_paths = [os.path.join(scene_dir, f'{b}.dat') for b in bands if os.path.exists(os.path.join(scene_dir, f'{b}.dat'))]
        band_paths = [os.path.join(scene_dir, f'{b}') for b in bands if os.path.exists(os.path.join(scene_dir, f'{b}'))]
        if not band_paths:
            # print(f'Skipping {scene_name}: no valid band files found.')
            continue

        # stacked_out_name = generate_scene_id_from_landsat_name(scene_name, extra=bands)
        stacked_out_name = scene_name
        stacked_out_dir = make_new_dir(output_folder, stacked_out_name)
        stacked_out_path = os.path.join(stacked_out_dir, stacked_out_name)

        counter += process_if_needed(
            stacked_out_path,
            replace_existing,
            stack_rasters,
            band_paths,
            stacked_out_path
        )
    return counter

def get_bands_from_stack(stack_path):
    """
    Open a stacked raster and return:
      - bands: list of 2D arrays (float)
      - band_names: list of strings matching your band_map
      - band_to_index: dict mapping band_name -> 1-based rasterio band index
      - profile: rasterio profile
    """
    import rasterio

    with rasterio.open(stack_path) as src:
        bands = [src.read(i + 1).astype(float) for i in range(src.count)]
        # We rely on the fact that band_names were set previously in the stack
        band_names = list(src.descriptions)
        if band_names is None or len(band_names) != src.count:
            # fallback: B1..Bn
            band_names = [f'B{i+1}' for i in range(src.count)]
        profile = src.profile.copy()
        band_to_index = {name: i + 1 for i, name in enumerate(band_names)}

    return bands, band_names, band_to_index, profile


def compute_composite_from_stack_and_save(stack_path, out_path, composite_name, band_map, sensor):
    """
    Compute a single composite or index (e.g., 'RGB', 'NDVI') from a stacked raster.

    Returns:
        np.ndarray: The computed composite (shape = [bands, height, width])
    """
    with rasterio.open(stack_path) as src:
        bands, band_names, band_to_index, profile = get_bands_from_stack(stack_path)

    tiny_offset = 1e-7

    red = bands[band_to_index[band_map[sensor]['R']] - 1]
    green = bands[band_to_index[band_map[sensor]['G']] - 1]
    blue = bands[band_to_index[band_map[sensor]['B']] - 1]
    nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
    swir = bands[band_to_index[band_map[sensor]['SWIR1']] - 1]

    match composite_name:
        case 'RGB':
            composite = np.stack([red, green, blue])

        case 'NDVI':
            composite = (nir - red) / (nir + red + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'EVI':
            composite = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'SAVI':
            L = 0.5
            composite = ((nir - red) * (1 + L)) / (nir + red + L)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDBI':
            composite = (swir - nir) / (swir + nir + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'MNDWI':
            composite = (green - swir) / (green + swir + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'FERRIC_IRON':
            composite = np.where(green == 0, 
                                 np.nan,
                                 red / green)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')
        
        case 'BAI':  # Burned Area Index
            composite = 1 / ((0.1 - blue)**2 + (0.06 - red)**2 + (nir - 0.3)**2)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'SI':  # Shadow Index
            composite = (nir - red) / (nir + red + green + tiny_offset)  # avoid div by 0
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'SR': # Simple ratio index
            composite = np.where(red == 0, 
                                 np.nan,
                                 nir / red)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDGI':  # Normalized Difference Greenness Index
            composite = (green - red) / (green + red + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDMI':  # Normalized Difference Moisture Index
            composite = (nir - swir) / (nir + swir + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'BSI':
            composite = ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'FeO':
            composite = np.where(blue == 0, 
                                 np.nan,
                                 red / blue)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDTI':
            composite = np.where(red + green == 0, 
                                 np.nan,
                                 (red - green) / (red + green))
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        # CMI is the same as NDBI
        # case 'CMI':  # Clay Mineral Index
        #     composite = (swir - nir) / (swir + nir + tiny_offset)
        #     composite = composite[np.newaxis, :, :]
        #     profile.update(dtype='float32')

        case _:
            raise ValueError(f"Unsupported composite/index: {composite_name}")

    # Update profile for output
    profile.update(
        driver='ENVI',
        count=composite.shape[0]
    )

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(composite)

    return composite


def build_composites_from_stacks(stacked_folder, output_folder, composites, band_map, replace_existing=False):
    """
    Build multiple composites or indices from stacked rasters in a folder.

    Parameters:
        stacked_folder (str): Path containing scene subfolders with stacked rasters.
        output_folder (str): Base path for output composites/indices.
        composites (list): List of composites/indices to compute (e.g., ['RGB', 'NDVI'])
        band_map (dict): Band mapping by sensor (global band_map).
        replace_existing (bool): Whether to overwrite existing outputs.
    """
    counter = 0
    for scene_dir in os.listdir(stacked_folder):
        scene_path = os.path.join(stacked_folder, scene_dir)
        if not os.path.isdir(scene_path):
            continue

        stack_files = [f for f in os.listdir(scene_path) if os.path.isfile(os.path.join(scene_path, f))]
        if not stack_files:
            print(f'No stack found in {scene_dir}')
            continue

        stack_path = os.path.join(scene_path, stack_files[0])
        sensor = extract_landsat_token(scene_dir)

        for composite in composites:
            out_dir = make_new_dir(output_folder, scene_dir)
            out_path = os.path.join(out_dir, f'{composite}')

            counter += process_if_needed(out_path, replace_existing, compute_composite_from_stack_and_save, stack_path, out_path, composite_name=composite, band_map=band_map, sensor=sensor)
    return counter

# --- Operational functions --- #

def process_if_needed(_out_path, _replace, process_func, *args, **kwargs):
    """
    Takes a function as input and runs it only if replace=True
    """
    if os.path.exists(_out_path) and not _replace:
        # print(f'Skipping processing of {_out_path}')
        return False
    process_func(*args, **kwargs)
    return True

def make_new_dir(root, new_dir_name):
    new_dir = os.path.join(root, new_dir_name)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

def get_all_files_in_dir_by_extension(dir_path, extension):
    dir_path = Path(dir_path)
    extension = extension.lstrip('.')
    return [str(p) for p in dir_path.glob(f'*.{extension}')]


# --- Naming functions --- #
def extract_scene_date(name: str) -> str | None:
    """Return first 8-digit date (YYYYMMDD) found in the scene name, or None."""
    parts = re.split(r'[_\-\s]', name)
    return next((p for p in parts if re.fullmatch(r'\d{8}', p)), None)

def extract_landsat_token(name: str) -> str | None:
    """Return the landsat token (e.g. LC08, LE07, LT05) found in the scene name, or None."""
    parts = re.split(r'[_\-\s]', name)
    return next((p for p in parts if re.match(r'^L.*\d{2}$', p, re.IGNORECASE)), None)

def generate_scene_id_from_landsat_name(fulllandsatname: str, extra: str | list | None = None) -> str:
    """
    Build an identifier "YYYYMMDD_Landsat" from a full scene name.
    If date or landsat token are missing, falls back to whichever is present,
    otherwise returns the original name. Optional `extra` is appended (list joined by '-').
    """
    date = extract_scene_date(fulllandsatname)
    landsat = extract_landsat_token(fulllandsatname)

    if date and landsat:
        base = f"{date}_{landsat}"
    elif date:
        base = date
    elif landsat:
        base = landsat
    else:
        base = fulllandsatname

    if extra is None:
        return base

    extra_str = '-'.join(map(str, extra)) if isinstance(extra, (list, tuple)) else str(extra)
    return f"{base}_{extra_str}"


class PlotUtils:
    def add_scalebar(ax, location='lower center', pad=0.05, colour='black', pixel_size=None):
        """
        Adds a scale bar to a GeoPandas/Matplotlib axis or raster image.
        
        Args:
            ax: Matplotlib axis
            location: Position of scalebar
            pad: Padding fraction from edges
            colour: Scalebar color
            pixel_size: If provided, uses this pixel size in meters for raster images.
                       If None, automatically detects from georeferenced data.
        """
        from matplotlib_scalebar.scalebar import ScaleBar
        
        # Case 1: Explicit pixel size provided (for raster images)
        if pixel_size is not None:
            scalebar = ScaleBar(
                dx=pixel_size,
                units='m',
                location=location,
                box_alpha=0,
                length_fraction=0.15,
                color=colour,
                pad=pad
            )
            ax.add_artist(scalebar)
            return
        
        # Case 2: Auto-detect from georeferenced data
        try:
            crs = getattr(ax, 'projection', None)
            if ax.get_xlim()[1] - ax.get_xlim()[0] > 360:  # projected (meters)
                scale = 1
                units = 'm'
            else:  # degrees
                scale = 111_000
                units = 'm'
        except Exception:
            scale = 1
            units = 'm'

        scalebar = ScaleBar(scale, units, location=location, box_alpha=0, length_fraction=0.25, color=colour, pad=pad)
        ax.add_artist(scalebar)


    @staticmethod
    def add_north_arrow(ax=None, color='black', size=None, location='lower right', pad=0.05):
        """
        Add a north arrow to a matplotlib plot.
        Pad specifies the fraction of axes to inset the arrow from edges.
        """
        if ax is None:
            ax = plt.gca()

        if size is None:
            bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
            size = 0.1 * (bbox.height / bbox.width)

        loc_map = {
            'upper right':  (1 - pad, 1 - pad),
            'upper left':   (pad + size, 1 - pad),
            'lower right':  (1 - pad, pad + size),
            'lower left':   (pad + size, pad + size)
        }
        x, y = loc_map.get(location, loc_map['upper right'])

        scale_factor = max(1, size)

        ax.annotate('N',
                    xy=(x, y),
                    xytext=(x, y - size),
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    ha='center', va='center',
                    fontsize=10*scale_factor,
                    color=color,
                    arrowprops=dict(facecolor=color, edgecolor=color, width=2*scale_factor, headwidth=10*scale_factor, headlength=10*scale_factor))

        return ax

    @staticmethod
    def add_geographic_labels(ax, raster_path, x_bounds=None, y_bounds=None, x_crop=(0, None), target_crs="EPSG:4326"):
        """
        Adds lat/lon labels to a raster plot in projected coordinates.
        
        Args:
            ax: Matplotlib axis
            raster_path: Path to the original raster file
            x_bounds: Tuple of (x_min, x_max) pixel bounds used to crop the raster
            y_bounds: Tuple of (y_min, y_max) pixel bounds used to crop the raster
            x_crop: Tuple of (left_crop, right_crop) if additional cropping was done.
                    e.g., [:,110:-110] would be (110, -110). Default (0, None) for no crop.
            crs: Source CRS of the raster (default "EPSG:32652")
            target_crs: Target CRS for labels (default "EPSG:4326" for lat/lon)
        """
        
        # Get the transform from the original raster
        with rasterio.open(raster_path) as src:
            if x_bounds is None:
                x_bounds = (0, src.width)
                print(x_bounds)
            if y_bounds is None:
                y_bounds = (0, src.height)
                print(y_bounds)
            window_transform = src.window_transform(((y_bounds[0], y_bounds[1]), (x_bounds[0], x_bounds[1])))
            crs = src.crs
                
        # Set up coordinate transformer
        transformer = pyproj.Transformer.from_crs(crs, target_crs, always_xy=True)
        
        # Get current tick positions (in pixels)
        y_ticks = ax.get_yticks()
        x_ticks = ax.get_xticks()
        
        # Handle x_crop offset
        x_offset = x_crop[0] if x_crop[0] is not None else 0
        
        # Convert pixel coordinates to geographic coordinates
        def pixel_to_geo(px, py):
            """Convert pixel coordinates to geographic coordinates"""
            utm_x, utm_y = window_transform * (px + x_offset, py)
            lon, lat = transformer.transform(utm_x, utm_y)
            return lon, lat
        
        # Get image dimensions
        img_height, img_width = ax.images[0].get_array().shape[:2]
        
        # Create geographic labels
        x_labels = []
        for x_px in x_ticks:
            if 0 <= x_px < img_width:
                lon, lat = pixel_to_geo(x_px, 0)
                x_labels.append(f"{lon:4.2f}°")
            else:
                x_labels.append("")
        
        y_labels = []
        for y_px in y_ticks:
            if 0 <= y_px < img_height:
                lon, lat = pixel_to_geo(0, y_px)
                y_labels.append(f"{lat:4.2f}°")
            else:
                y_labels.append("")
        
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Longitude", fontsize=16)
        ax.set_ylabel("Latitude", fontsize=16)