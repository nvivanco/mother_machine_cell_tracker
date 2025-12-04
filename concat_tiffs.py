import os
import re
import numpy as np
from tifffile import imread, imwrite

"""
This functinon takes in the path to a directory with a set of tiff files and stitches them together so that they play together. Tiff files are 4D (TCXY) and need
to be approx the same size in terms of height. Function does some padding but may break for bigger tiff files 
"""

# used to sort files in numerical order 
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

#change output path as needed 
def concat_hyperstack_tifs(directory, output_path="combined.tif"):
    # files is list of files in directory provided 
    files = [f for f in os.listdir(directory)
             if f.lower().endswith((".tif", ".tiff"))]
    if not files:
        raise ValueError("No TIFF files found.")
    
    #sorts files so they appear in order, in our case it would oprganize by FOV     
    files.sort(key=natural_sort_key)

    # Load stacks
    imgs = []
    for f in files:
        arr = imread(os.path.join(directory, f))
        print(f"{f} shape: {arr.shape}")
        imgs.append(arr)

    # Expect shape (T, C, H, W)
    # Ensure consistent T, C
    t_all = [a.shape[0] for a in imgs]
    c_all = [a.shape[1] for a in imgs]

    if len(set(t_all)) != 1:
        raise ValueError(f"Inconsistent Z depths: {t_all}")
    if len(set(c_all)) != 1:
        raise ValueError(f"Inconsistent channel counts: {c_all}")

    # Crop to minimally shared height for tif files that are not the same pixel height (x should not matter since we're stiching horizontally)
    heights = [a.shape[2] for a in imgs]
    min_h = min(heights)

    imgs_cropped = [a[:, :, :min_h, :] for a in imgs]

    # Concatenate horizontally along width axis (-1) using numpy array function
    combined = np.concatenate(imgs_cropped, axis=3)

    # Save output
    imwrite(output_path, combined)
    print(f"Saved combined hyperstack to {output_path}")
    print(f"Final shape: {combined.shape}")

    return combined
