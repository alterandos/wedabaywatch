from src import composite_analysis
import os
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image

import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from skimage.morphology import remove_small_objects
from scipy.ndimage import label

roi_sources = {
    "Cloud": [("LC08_L2SP_109060_20230506_20230509_02_T1", "2023_x4_v2_SHPs/Cloud.shp")],
    "Jungle": [("LC08_L2SP_109060_20230506_20230509_02_T1", "2023_x4_v2_SHPs/Jungle.shp"),
              ("LC08_L2SP_109060_20160603_20200907_02_T1", "2016_x2_SHPs/Burnt_Jungle.shp")],
    "Mining": [("LC08_L2SP_109060_20230506_20230509_02_T1", "2023_x4_v2_SHPs/Mining.shp")],
    "Water": [("LC08_L2SP_109060_20230506_20230509_02_T1", "2023_x4_v2_SHPs/Water.shp"),
        ("LC08_L2SP_109060_20190204_20200829_02_T1", "2019_x1_SHPs/Ocean.shp"),
        ("LC08_L2SP_109060_20160603_20200907_02_T1", "2016_x2_SHPs/Ocean.shp")
    ],
    #"Burnt": [("stacked/2016", "ROIs/2016_x2_SHPs/Burnt_Forest.shp") ]
}


def get_cloud_mask_from_stacked(img_name, img_folder_path = "data/stacked/"):
    """ Gets cloud mask from the image in data/stacked folder"""
    with rasterio.open(img_folder_path + img_name + "/" + img_name) as src:
        img_data = src.read()

    qa = img_data[-1]
    nodata = src.nodata

    cloud_shadow=True

    cirrus=False

    cloud_mask = (
        (
            (qa & composite_analysis.QA_BITS['cloud'])
            | (qa & composite_analysis.QA_BITS['cloud_shadow'] & cloud_shadow)
            | (qa & composite_analysis.QA_BITS['cirrus'] & cirrus)
        ) > 0
    )

    na_mask = (qa == 1)

    if nodata is not None:
        cloud_mask[qa == nodata] = True  # treat nodata as "ignore"

    return cloud_mask, na_mask


def extract_pixels_from_roi(img_name, 
                            roi_name, 
                            roi_folder_path = "output/rois/classification_training_rois/",
                            img_folder_path = "data/stacked/",
                            cloud_masking = True
                            ):
    """Extract pixels inside ROI polygons for a given image and shapefile."""
    with rasterio.open(img_folder_path + img_name + "/" + img_name) as src:
        img_data = src.read()
        transform = src.transform
        

    roi_gdf = gpd.read_file(roi_folder_path + roi_name)
    mask = rasterio.features.geometry_mask(
        roi_gdf.geometry,
        out_shape=(img_data.shape[1], img_data.shape[2]),
        transform=transform,
        invert=True
    )
    
    cloud_mask, _ = get_cloud_mask_from_stacked(img_name, img_folder_path)

    ys, xs = np.where(mask & ~(cloud_mask & cloud_masking) )
    pixels = img_data[:, ys, xs].T
    return pixels

def extract_and_train(
    roi_sources, 
    roi_folder_path = "data/training_rois/",
    img_folder_path = "data/stacked/",
    cloud_masking = True
):
    """ Extracts all classes from given ROI dict, and trains classifier. """
    classes = list(roi_sources.keys())

    X_list, y_list = [], []

    for class_id, class_name in enumerate(classes):
        for img_path, shp_path in roi_sources[class_name]:
            pixels = extract_pixels_from_roi(img_path, shp_path, 
                                             roi_folder_path, img_folder_path,
                                             cloud_masking)
            X_list.append(pixels)
            y_list.append(np.full(len(pixels), class_id))

    # -------------------------------
    # Stack into training arrays             
    # -------------------------------
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    # -------------------------------
    # Train classifier
    # -------------------------------
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)

    return clf


def clean_small_regions(arr, min_size=50, connectivity=2):
    """
    Removes connected regions smaller than `min_size` for each unique value in arr.
    
    Parameters
    ----------
    arr : 2D numpy array (numeric labels, e.g. -1..3)
    min_size : int
        Minimum region size (in pixels) to keep.
    connectivity : int
        1 for 4-connectivity, 2 for 8-connectivity.
    """
    out = np.copy(arr)
    for val in np.unique(arr):
        mask = arr == val
        labeled, n = label(mask, structure=np.ones((3,3)) if connectivity == 2 else None)
        if n == 0:
            continue
        counts = np.bincount(labeled.ravel())
        remove_mask = np.isin(labeled, np.where(counts < min_size)[0])
        out[remove_mask] = np.median(arr)  # or another fill value, e.g. np.nan
    return out

def classify_img(clf,
              img_name, 
              img_folder_path = "data/stacked/",
              classified_folder_path = "output/classified/",
              plot = False,
              cloud_masking = True):
    """ Apply trained classifier to specified image"""

    # -------------------------------
    # 1. Open image
    # -------------------------------

    img_path = img_folder_path + img_name + "/" + img_name
    with rasterio.open(img_path) as src:
        img = src.read()
        profile = src.profile
        rgb = reshape_as_image(img)[:, :, :3]  # for display
        transform = src.transform
    
    # -------------------------------
    # 2. Classify image
    # -------------------------------
    h, w = img.shape[1], img.shape[2]
    X = img.reshape(img.shape[0], -1).T
    y_pred = clf.predict(X)
    classified = y_pred.reshape(h, w)
    
    # -------------------------------
    # 3. Mask cloud
    # ------------------------------- 

    cloud_mask, na_mask = get_cloud_mask_from_stacked(img_name, img_folder_path)

    if cloud_masking:
        classified[cloud_mask] = 0

    classified[na_mask] = -2

    if plot:    
        plt.figure(figsize=(8, 6))
        plt.imshow(classified, cmap="tab20")
        plt.title("Raw classification")
        plt.axis("off")
        plt.show()

    # -------------------------------
    # 4. Smooth result
    # ------------------------------- 
    smoothed = clean_small_regions(classified)
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(smoothed, cmap="tab20")
        plt.title("Smoothed classification")
        plt.axis("off")
        plt.show()
    
    # -------------------------------
    # 4. Save smoothed result
    # -------------------------------
    out_path = classified_folder_path + img_name + "_classified.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(smoothed.astype(np.uint8), 1)

    return smoothed

def train_and_classify(roi_sources,                             
                       roi_folder_path = "data/training_rois/",
                       img_folder_path = "data/stacked/",
                       classified_folder_path = "output/classified/",
                       cloud_masking = True,
                       plot = False,
                       replace_existing = True
                       ):
    """ Combine functions to build full classification pipeline """

    if replace_existing == False:
        return 0

    # Make sure each path ends in "/"
    if roi_folder_path[-1] != "/":
        roi_folder_path = roi_folder_path + "/"

    if img_folder_path[-1] != "/":
        img_folder_path = img_folder_path + "/"

    if classified_folder_path[-1] != "/":
        classified_folder_path = classified_folder_path + "/"
            
    # Train classifier
    clf = extract_and_train(roi_sources, 
                            img_folder_path = img_folder_path, 
                            roi_folder_path = roi_folder_path,
                            cloud_masking = cloud_masking)

    # Classify all images
    for img_name in os.listdir(img_folder_path):
        classify_img(clf,
                     img_name, 
                     img_folder_path = img_folder_path,
                     classified_folder_path = classified_folder_path,
                     plot = plot, 
                     cloud_masking = cloud_masking)
    
    return len(os.listdir(img_folder_path))