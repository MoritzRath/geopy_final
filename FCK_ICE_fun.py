# In this file I gather functions, that I'll use in the processing.
import os
import ee
import geemap
import tempfile
import re
import shutil
from arosics import COREG_LOCAL
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from pathlib import Path
from datetime import datetime
from collections import defaultdict


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
            
            coreg = None
            crl_afterCORR = None
        else:
            continue

def renameTifs(path, out_folder=None, move=False, recursive=True):
    """
    This function takes a folder and renames the coregisterd tifs.
    """

    src = Path(path)

    if out_folder is None:
        dst = src.parent / f"{src.name}_coregistered"
    else:
        dst = Path(out_folder)

    pattern = re.compile(r"(\d{8})__shifted_to__(\d{8})", re.IGNORECASE)
    records = []

    iterator = src.rglob("*") if recursive else src.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in (".tif")]
    
    dst.mkdir(parents=True, exist_ok=True)
    
    for f in files:
        m = pattern.search(f.stem)
        if not m:
            continue       

        d1 = datetime.strptime(m.group(1), "%Y%m%d")
        d2 = datetime.strptime(m.group(2), "%Y%m%d")
        mid_date = d1+ (d2 -d1) / 2
        delta_days = (d2-d1).days

        #mid_ordinal = (d1.toordinal() + d2.toordinal()) // 2
        #mid_date = datetime.fromordinal(mid_ordinal).strftime("%Y%m%d")

        target = dst / f"{mid_date:%Y%m%d}_coregistered.tif"

        if move:
            shutil.move(str(f), str(target))
        else:
            shutil.copy2(f, target)

        records.append({
            "src": str(f),
            "dst": str(target),
            "date1": d1,
            "date2": d2,
            "mid_date": mid_date,
            "delta_days": delta_days,
        })

    return pd.DataFrame(records)
"""
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
"""

def shift_to_vel(df, pixel_size=1):
    df = df.copy()
    df["mid_date"] = pd.to_datetime(df["mid_date"])

    velocities = []
    for r in df.itertuples(index=False):
        with rasterio.open(r.dst) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata if src.nodata is not None else np.nan
        
        arr[arr == nodata] = np.nan

        displacement_m = np.nanmean(np.abs(arr)) * pixel_size
        velocity_m_per_day = displacement_m / r.delta_days
        velocities.append((row.mid_date, velocity_m_per_day))

    s = pd.Series(
        {date: vel for date, vel in velocities}
    ).sort_index()

    daily_index = pd.date_range(s.index.min().floor("D"), s.index.max().ceil("D"), freq="D")
    daily_velocity = s.reindex(daily_index). interpolate("time").ffill().bffill()

    return daily_velocity

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

def get_dates(path):
    """
    This function returns the dates read from the file names
    """
    dates = []

    for tif in path:
        name = Path(tif).name
        d_str = name[:8]
        dates.append(datetime.strptime(d_str, "%Y%m%d"))
    
    return dates


def compute_export_means(
        masked_arrays,
        dates,
        template,
        out_dir,
        base_name="mean_ice_velocity"):
    """
    this function computes annual and monthly means of the rasters and writes them as tif to the disk.
    """
    if len(masked_arrays) != len(dates):
        raise ValueError("masked arrays and dates must have the same length")
    
    stack = np.stack([np.squeeze(arr) for arr in masked_arrays], axis=0).astype("float32")
    nodata = template["nodata"]
    stack[stack == nodata] = np.nan

    by_year = defaultdict(list)
    by_month = defaultdict(list)

    for i, d in enumerate(dates):
        by_year[d.year].append(i)
        by_month[(d.year, d.month)].append(i)
    
    annual_means = {y: np.nanmean(stack[i], axis=0) for y,i in by_year.items()}
    monthly_means = {(y,m): np.nanmean(stack[i], axis=0) for (y,m),i in by_month.items()}

    profile = {
        "driver": "GTiff",
        "height": template["height"],
        "width": template["width"],
        "count": 1,
        "dtype": "float32",
        "crs": template["crs"],
        "transform": template["transform"],
        "nodata": np.nan,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for y, arr in annual_means.items():
        out_path = out_dir / f"{y}_{base_name}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr.astype("float32"),1)

    for (y, m), arr in monthly_means.items():
        out_path = out_dir / f"{y}_{m:02d}_{base_name}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr.astype("float32"),1)
    
    return annual_means, monthly_means

def single_mean_abs(
        masked_arrays,
        dates, 
        template=None
):
    """
    This function calculates the mean of a single shift raster and its absolute and stores it in a dictionaries
    """
    if len(masked_arrays) != len(dates):
        raise ValueError("masked arrays and dates must have the same length")
    
    nodata = None if template is None else template.get("nodata", None)
    
    single_abs = {}
    single_means = {}

    for arr, d in zip(masked_arrays, dates):
        key = (d.year, d.month, d.day)
        a = np.squeeze(arr).astype("float32")

        if nodata is not None:
            a[a== nodata] = np.nan
        
        abs_a = np.abs(a)

        single_abs[key] = abs_a
        single_means[key] = float(np.nanmean(abs_a))
    
    return single_abs, single_means
