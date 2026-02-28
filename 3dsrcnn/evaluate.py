import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import Slider

import config
import metrics
import preprocessing

def visualize_comparison(ground_truth, low_resolution, upsampled, prediction, psnr_value, ssim_value):
    datacubes = [ground_truth, low_resolution, upsampled, prediction]
    num_bands = ground_truth.shape[2]

    vmin_per_band, vmax_per_band = [], []

    for band in range(num_bands):
        vmin_per_band.append(min(cube[:, :, band].min() for cube in datacubes))
        vmax_per_band.append(max(cube[:, :, band].max() for cube in datacubes))

    vmin_per_band = np.array(vmin_per_band)
    vmax_per_band = np.array(vmax_per_band)

    displayed_band = 0

    fig, ax = plt.subplots(ncols=4, sharex=True, sharey=True)

    # Display low-resolution image
    low_res_slice = low_resolution[:, :, displayed_band]
    low_res_im = ax[0].imshow(low_res_slice, cmap='gray', vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])
    ax[0].set_title('LR Patch (Blur)')

    upsampled_slice = upsampled[:, :, displayed_band]
    upsampled_im = ax[1].imshow(upsampled_slice, cmap='gray', vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])
    ax[1].set_title('LR Patch (Bicubic Upsampling)')

    prediction_slice = prediction[:, :, displayed_band]
    prediction_im = ax[2].imshow(prediction_slice, cmap='gray', vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])
    ax[2].set_title(f'Enhanced Patch (PSNR: {psnr_value:.2f}, Average SSIM: {ssim_value:.2f})')

    ground_truth_slice = ground_truth[:, :, displayed_band]
    ground_truth_im = ax[3].imshow(ground_truth_slice, cmap='gray', vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])
    ax[3].set_title('Ground Truth')

    ax_band_slider = plt.axes([0.2, 0.15, 0.65, 0.03])
    band_slider = Slider(ax_band_slider, 'Band Num', 1, ground_truth.shape[2], valinit=displayed_band + 1, valstep=1)

    def update(val):
        nonlocal displayed_band
        displayed_band = int(band_slider.val) - 1

        new_low_res_slice = low_resolution[:, :, displayed_band]
        low_res_im.set_data(new_low_res_slice)
        low_res_im.set_clim(vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])

        new_upsampled_slice = upsampled[:, :, displayed_band]
        upsampled_im.set_data(new_upsampled_slice)
        upsampled_im.set_clim(vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])

        prediction_slice = prediction[:, :, displayed_band]
        prediction_im.set_data(prediction_slice)
        prediction_im.set_clim(vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])

        ground_truth_slice = ground_truth[:, :, displayed_band]
        ground_truth_im.set_data(ground_truth_slice)
        ground_truth_im.set_clim(vmin=vmin_per_band[displayed_band], vmax=vmax_per_band[displayed_band])

        fig.canvas.draw_idle()

    def on_press(event):
        if event.key == 'left' or event.key == 'a':
            new_slider_val = max(band_slider.valmin, band_slider.val - 1)
            band_slider.set_val(new_slider_val)

        elif event.key == 'right' or event.key == 'd':
            new_slider_val = min(band_slider.valmax, band_slider.val + 1)
            band_slider.set_val(new_slider_val)

    band_slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()

# ========== Load Trained Model ==========
model = tf.keras.models.load_model(
    './3dsrcnn/models/3d_srcnn_2x.keras', custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
)

# ========== Load Dataset ==========
datacube_file = 'indian_pines.mat'
datacube_key = 'indian_pines'
datacube_file = 'cuprite.mat'
datacube_key = 'X'
datacube_file = 'salinas.mat'
datacube_key = 'salinas'
datacube = preprocessing.load_datacube(config.DATA_FOLDER, datacube_file, datacube_key)

# Indices of top-left of patch, list all coordinates that will be visualized
patch_coordinates = [(25, 25)]

ssim_values = []
psnr_values = []

for x, y in patch_coordinates:
    # ========== Extract Patch from Datacube (Treat as HR data) ==========
    patch_size = config.PATCH_SIZE
    high_res_patch = datacube[x : x + patch_size, y : y + patch_size, :]
    print(f'Original HR Patch Shape: {high_res_patch.shape}')

    # Resize patch if smaller than 64x64
    if high_res_patch.shape[0] != patch_size or high_res_patch.shape[1] != patch_size:
        num_bands = high_res_patch.shape[2]
        resized_high_res_patch = np.zeros((patch_size, patch_size, num_bands), dtype=high_res_patch.dtype)

        for band in range(num_bands):
            resized_high_res_patch[:, :, band] = cv2.resize(
                high_res_patch[:, :, band],
                (patch_size, patch_size),
                interpolation=cv2.INTER_CUBIC
            )

        high_res_patch = resized_high_res_patch
        print(f'Resized HR Patch Shape: {high_res_patch.shape}')

    # ========== Generate LR Version of patch ==========
    scale_factor = config.SCALE_FACTOR
    low_res_patch = preprocessing.generate_low_res_patch(high_res_patch, scale_factor)
    upsampled_patch = preprocessing.upsample_low_res_patch(low_res_patch, scale_factor)

    # ========== Prepare Input for Model ==========
    # Model expects shape: (batch_size, height, width, channels, 1)
    upsampled_patch_input = np.expand_dims(upsampled_patch, axis=0)
    upsampled_patch_input = np.expand_dims(upsampled_patch_input, axis=-1)

    # ========== Enhance Patch Using Model ==========
    enhanced_patch = model.predict(upsampled_patch_input)
    enhanced_patch = np.squeeze(enhanced_patch)

    # ========== Compute Metrics ==========
    psnr_value = metrics.compute_psnr(high_res_patch, enhanced_patch)
    ssim_value = metrics.compute_average_ssim(high_res_patch, enhanced_patch)

    psnr_values.append(psnr_value)
    ssim_values.append(ssim_value)

    print(f'PSNR: {psnr_value:.5f} dB')
    print(f'SSIM: {ssim_value:.5f}')

    # ========== Visualization ==========
    visualize_comparison(high_res_patch, low_res_patch, upsampled_patch, enhanced_patch, psnr_value, ssim_value)
