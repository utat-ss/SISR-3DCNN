import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(ground_truth, prediction):
    return psnr(ground_truth, prediction)

def compute_average_ssim(ground_truth, prediction):
    num_bands = ground_truth.shape[2]
    ssim_values = []

    for band in range(num_bands):
        ground_truth_band = ground_truth[:, :, band]
        prediction_band = prediction[:, :, band]
        data_range = ground_truth_band.max() - ground_truth_band.min()

        if data_range == 0:
            if np.allclose(ground_truth_band, prediction_band):
                ssim_band_value = 1.0
            else:
                ssim_band_value = 0.0
        else:
            ssim_band_value = ssim(
                ground_truth_band,
                prediction_band,
                data_range=data_range
            )

        ssim_values.append(ssim_band_value)

    return np.mean(ssim_values)