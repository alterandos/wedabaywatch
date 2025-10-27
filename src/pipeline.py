import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping, Polygon

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

# Band map
band_map = {
    'LC09': {  # Landsat 8
        'AEROSOL': 'B1',
        'B': 'B2',
        'G': 'B3',
        'R': 'B4',
        'NIR': 'B5',
        'SWIR1': 'B6',
        'SWIR2': 'B7',
        'TIR1': 'B10',
        'TIR2': 'B11'
    },
    'LC08': {  # Landsat 8
        'AEROSOL': 'B1',
        'B': 'B2',
        'G': 'B3',
        'R': 'B4',
        'NIR': 'B5',
        'SWIR1': 'B6',
        'SWIR2': 'B7',
        'TIR1': 'B10',
        'TIR2': 'B11'
    },
    'LE07': {  # Landsat 7
        'B': 'B1',
        'G': 'B2',
        'R': 'B3',
        'NIR': 'B4',
        'SWIR1': 'B5',
        'TIR': 'B6',
        'SWIR2': 'B7'
    }
}

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
            # r = bands[band_to_index[band_map[sensor]['R']] - 1]
            # g = bands[band_to_index[band_map[sensor]['G']] - 1]
            # b = bands[band_to_index[band_map[sensor]['B']] - 1]
            composite = np.stack([red, green, blue])

        case 'NDVI':
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            # red = bands[band_to_index[band_map[sensor]['R']] - 1]
            composite = np.where(
                (nir + red) == 0, np.nan,
                (nir - red) / (nir + red)
            )
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'EVI':
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            # r = bands[band_to_index[band_map[sensor]['R']] - 1]
            # b = bands[band_to_index[band_map[sensor]['B']] - 1]
            composite = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'SAVI':
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            # r = bands[band_to_index[band_map[sensor]['R']] - 1]
            L = 0.5
            composite = ((nir - red) * (1 + L)) / (nir + red + L)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDBI':
            # swir = bands[band_to_index[band_map[sensor]['SWIR1']] - 1]
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            composite = np.where(
                (swir + nir) == 0, np.nan,
                (swir - nir) / (swir + nir)
            )
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'MNDWI':
            # green = bands[band_to_index[band_map[sensor]['G']] - 1]
            # swir = bands[band_to_index[band_map[sensor]['SWIR1']] - 1]
            composite = np.where(
                (green + swir) == 0, np.nan,
                composite = (green - swir) / (green + swir)
            )
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'FERRIC_IRON':
            # red = bands[band_to_index[band_map[sensor]['R']] - 1]
            # green = bands[band_to_index[band_map[sensor]['G']] - 1]
            composite = red/(green+1e-7)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')
        
        case 'BAI':  # Burned Area Index
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            # red = bands[band_to_index[band_map[sensor]['R']] - 1]
            # blue = bands[band_to_index[band_map[sensor]['B']] - 1]
            composite = 1 / ((0.1 - blue)**2 + (0.06 - red)**2 + (nir - 0.3)**2)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'SI':  # Shadow Index
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            # red = bands[band_to_index[band_map[sensor]['R']] - 1]
            # green = bands[band_to_index[band_map[sensor]['G']] - 1]
            composite = (nir - red) / (nir + red + green + tiny_offset)  # avoid div by 0
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDGI':  # Normalized Difference Greenness Index
            # green = bands[band_to_index[band_map[sensor]['G']] - 1]
            # red = bands[band_to_index[band_map[sensor]['R']] - 1]
            composite = (green - red) / (green + red + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'NDMI':  # Normalized Difference Moisture Index
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            # swir = bands[band_to_index[band_map[sensor]['SWIR1']] - 1]
            composite = (nir - swir) / (nir + swir + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

        case 'CMI':  # Clay Mineral Index
            # swir = bands[band_to_index[band_map[sensor]['SWIR1']] - 1]
            # nir = bands[band_to_index[band_map[sensor]['NIR']] - 1]
            composite = (swir - nir) / (swir + nir + tiny_offset)
            composite = composite[np.newaxis, :, :]
            profile.update(dtype='float32')

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