# In this file I gather functions, that I'll use in the processing.
import os
import ee
import geemap
import tempfile
import re
import shutil
from arosics import COREG_LOCAL
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from pathlib import Path
from datetime import datetime


def getS1date(img):
    index = img.get('system:index').getInfo()
    date = index.split('_')[4][:8]
    file = f"{date}.tif"
    return file

def coregisterS1(s1, region, kwargs):
    """
    This function coregisters two consecutive Sentinel-1 iteratively images from an GEE ImageCollection.
    The function then stores the coregistered images in a given directory.
    Following variables:

    s1:             Image collection of S1-Scenes obtained via GEE
    region:         regional extent in which the coregistration will be performed
    kwargs:         parameters for coregistering
    """

    # fix regional extent
    regionEXP = ee.FeatureCollection(region).geometry()
    regionEXP = regionEXP.transform('EPSG:4326', maxError=1)

    # Transform S1-Scenes to list
    collection = s1.toList(s1.size())
    n_images = s1.size().getInfo()

    # Initiate temp-directory
    temp_dir = tempfile.mkdtemp()

    for i in range(n_images-1):
        img1 = ee.Image(collection.get(i))
        img2 = ee.Image(collection.get(i+1))

        file1 = getS1date(img1)
        file2 = getS1date(img2)

        path_img1 = os.path.join(temp_dir, file1)
        path_img2 = os.path.join(temp_dir, file2)

        geemap.ee_export_image(img1, filename=path_img1, scale=10,
                               region=regionEXP, file_per_band=False)
        geemap.ee_export_image(img2, filename=path_img2, scale=10,
                               region=regionEXP, file_per_band=False)
        
        if os.path.exists(path_img1) and os.path.exists(path_img2):
            coreg = COREG_LOCAL(path_img2, path_img1, **kwargs)
            coreg.correct_shifts()
            crl_afterCORR = COREG_LOCAL(path_img2, coreg.path_out, **kwargs)
            #crl_afterCORR.view_CoRegPoints(figsize=(15,15), backgroundIm='ref')
            coreg = None
            crl_afterCORR = None
        else:
            continue

def renameTifs(path, out_folder=None, move=False, recursive=True):
    """
    This function takes a folder and renames the coregisterd tifs. Eventually.
    """

    src = Path(path)

    if out_folder is None:
        dst = src.parent / f"{src.name}_coregistered"
    else:
        dst = Path(out_folder)

    pattern = re.compile(r"(\d{8})__shifted_to__(\d{8})", re.IGNORECASE)
    iterator = src.rglob("*") if recursive else src.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in (".tif")]
    
    ops = []
    unmatched = []

    for f in files:
        if dst in f.parents:
            continue

        m = pattern.search(f.stem)
        if not m:
            unmatched.append(f.name)
            continue

        d1 = datetime.strptime(m.group(1), "%Y%m%d").date()
        d2 = datetime.strptime(m.group(2), "%Y%m%d").date()

        mid_ordinal = (d1.toordinal() + d2.toordinal()) // 2
        mid_date = datetime.fromordinal(mid_ordinal).strftime("%Y%m%d")

        target = dst / f"{mid_date}_coregistered.tif"

        if target.exists():
            i = 1
            while True:
                candidate = dst / f"{mid_date}_coregistered_{i:02d}.tif"
                if not candidate.exists():
                    target = candidate
                    break
                i += 1
        ops.append((f, target))

    print(f"Source folder: {src}")
    print(f"Found tif/tiff files: {len(files)}")
    print(f"Matched rename pattern: {len(ops)}")
    if unmatched:
        print(f"Unmatched files (first 10): {unmatched[:10]}")

    
    dst.mkdir(parents=True, exist_ok=True)

    for old, new in ops:
        if move:
            shutil.move(str(old), str(new))
        else:
            shutil.copy2(old, new)

    print(f"Written files: {len(ops)} to {dst}")   
    return ops

def loadTifs(filepath):
    """
    This function takes a folder path an reads all the GTIF-files in that folder into a list and returns it.
    """

    path = sorted(Path(filepath).glob('*.tif'))
    return [rasterio.open(f) for f in path]



def maskTif(tif, 
            region_shp, 
            glacier_shp, 
            crop=False, 
            nodata=None, 
            invert_glacier=False,
            template=None):
    """
    Ideally, this function will take a tif and two shapefiles and mask out everything outside the area
    """
    
    if nodata is None:
        nodata = tif.nodata if tif.nodata is not None else 0

    region_shp = region_shp.to_crs(tif.crs)
    glacier_shp = glacier_shp.to_crs(tif.crs)

    arr_region, tr_region = mask(
        tif, 
        region_shp.geometry,
        crop=crop,
        filled=True,
        nodata=nodata
    )

    profile = tif.profile.copy()
    profile.update(
        height=arr_region.shape[1],
        width=arr_region.shape[2],
        transform=tr_region,
        nodata=nodata
    )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as tmp_ds:
            tmp_ds.write(arr_region)
            arr_final, tr_final = mask(
                tmp_ds,
                glacier_shp.geometry,
                crop=crop,
                filled=True,
                nodata=nodata,
                invert=invert_glacier
            )
    
    if template is not None:
        dst = np.full(
            (arr_final.shape[0], template["height"], template["width"]),
            nodata,
            dtype=arr_final.dtype
        )

        for b in range(arr_final.shape[0]):
            reproject(
                source=arr_final[b],
                destination=dst[b],
                src_transform=tr_final,
                src_crs=tif.crs,
                dst_transform=template["transform"],
                dst_crs=template["crs"],
                scr_nodata=nodata,
                dst_nodata=nodata,
                resampling=Resampling.nearest
            )

        return dst, template["transform"]

    return arr_final, tr_final

def maskTif_loop(
        tif_list,
        region_shp,
        glacier_shp,
        crop=False,
        nodata=None,
        invert_glacier=False
):
    
    masked_arrays = []
    template = None

    for tif in tif_list:
        arr, tr = maskTif(
            tif, 
            region_shp,
            glacier_shp,
            crop=crop,
            nodata=nodata,
            invert_glacier=invert_glacier
        )
    
        if template is None:
            nd = nodata if nodata is not None else (tif.nodata if tif.nodata is not None else 0)
            template = {
                "crs": tif.crs,
                "transform": tr,
                "height": arr.shape[1],
                "width": arr.shape[2],
                "nodata": nd,
                "dtype": arr.dtype,
            }
            masked_arrays.append(arr)
            continue

        dst = np.full(
            (arr.shape[0], template["height"], template["width"]),
            template["nodata"],
            dtype=template["dtype"],
        )

        for b in range(arr.shape[0]):
            reproject(
                source=arr[b],
                destination=dst[b],
                src_transform=tr,
                src_crs=tif.crs,
                dst_transform=template["transform"],
                dst_crs=template["crs"],
                src_nodata=template["nodata"],
                dst_nodata=template["nodata"],
                resampling=Resampling.nearest,
            )
    
        masked_arrays.append(dst)

    return masked_arrays, template

