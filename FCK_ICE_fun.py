# In this file I gather functions, that I'll use in the processing.
import os
import ee
import geemap
import tempfile
from arosics import COREG_LOCAL
import rasterio
from rasterio.mask import mask
from pathlib import Path


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

def renameTifs(path):
    """
    This function takes a folder and renames the coregisterd tifs. Eventually.
    """

    """
    alles string
    erstmal splitte ich by seperator
    und dann bekomm ich ne liste zurück
    und dann inidizier ich 0 und last
    ODER
    ich benuzte wie hieß das nochmal
    anna kann sich keine sachen merken und ich schreib das nicht auf
    find und rfind
    find ist für den ersten abschnitt
    und rfind für den letzten"""

def loadTifs(filepath):
    """
    This function takes a folder path an reads all the GTIF-files in that folder into a list and returns it.
    """

    path = sorted(Path(filepath).glob('*.tif'))
    return [rasterio.open(f) for f in path]



def maskTif(tif, shapefile):
    """
    Ideally, this function will take a tif and a shapefile and mask out everything outside the area
    """
    shapefile = shapefile.to_crs(tif.crs)

    masked_tif, _ = mask(tif, shapefile.geometry, filled=True)
    return masked_tif