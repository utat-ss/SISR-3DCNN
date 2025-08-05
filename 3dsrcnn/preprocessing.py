from pathlib import Path

import cv2
import numpy as np
import scipy.io

from config import PATCH_BANDS, BLUR_KERNEL_SIZE, BLUR_SIGMA

def load_datacube(directory, filename, key):
    matlab_data = scipy.io.loadmat(Path(directory, filename))

    # To find the key, print matlab_data and find the hyperspectral data variable
    datacube = matlab_data[key]

    # Convert to float and normalize to range [0, 1]
    datacube = datacube.astype(np.float32)
    datacube /= np.max(datacube)

    # The AVIRIS sensor (used for indian_pines, cuprite, salinas) captures data
    # across 224 bands, but the Indian Pines dataset has 4 bands removed (indices below)
    # For consistency, any AVIRIS datacube with all 224 bands should have the
    # same bands removed as in the Indian Pines dataset
    if datacube.shape[2] == 224:
        removed_band_indices = [0, 32, 96, 160]
        datacube = np.delete(datacube, removed_band_indices, axis=2)

    return datacube

def extract_patches(datacube, patch_size, stride):
    patches = []
    H, W, _ = datacube.shape

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = datacube[i : i + patch_size, j : j + patch_size, :]
            patches.append(patch)

    return np.array(patches)

def generate_low_res_patch(high_res_patch, scale_factor, blur=True, blur_kernel_size=BLUR_KERNEL_SIZE, blur_sigma=BLUR_SIGMA):
    h, w, _ = high_res_patch.shape
    new_h, new_w = h // scale_factor, w // scale_factor

    # Apply Gaussian blur bandwise
    blurred_patch = high_res_patch

    if blur:
        blurred_patch = np.zeros_like(high_res_patch)

        for band in range(high_res_patch.shape[2]):
            blurred_patch[:, :, band] = cv2.GaussianBlur(
                high_res_patch[:, :, band],
                ksize=(blur_kernel_size, blur_kernel_size),
                sigmaX=blur_sigma,
                borderType=cv2.BORDER_REFLECT
            )

    # Downsample (acquire low-res image) + upsample (match original dimensions)
    downscaled_patch = cv2.resize(blurred_patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    low_res_patch = cv2.resize(downscaled_patch, (w, h), interpolation=cv2.INTER_CUBIC)

    return low_res_patch
