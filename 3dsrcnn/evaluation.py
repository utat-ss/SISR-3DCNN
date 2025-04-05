# stdlib
import math

# external
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model

# 1. Load the Trained Model
model = load_model(
    "srcnn_model.h5", custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)

# 2. Load and Preprocess the Hyperspectral Dataset
data = scipy.io.loadmat("Indian_pines.mat")
hs_data = data["indian_pines"]

hs_data = hs_data.astype(np.float32)
hs_data /= np.max(hs_data)

# 3. Extract a Patch from the HR Data
patch_size = 64
# Choose indices such that we extract a patch. Here we choose (x, y) = (100, 100).
# adjust these indices based on the dimensions of hs_data.
x, y = 100, 100
hr_patch = hs_data[x : x + patch_size, y : y + patch_size, :]
print("Original HR Patch shape:", hr_patch.shape)

# If the extracted patch is smaller than 64x64, resize it to (64,64) for consistency.
if hr_patch.shape[0] != patch_size or hr_patch.shape[1] != patch_size:
    num_bands = hr_patch.shape[2]
    resized_patch = np.zeros((patch_size, patch_size, num_bands), dtype=hr_patch.dtype)
    for band in range(num_bands):
        resized_patch[:, :, band] = cv2.resize(
            hr_patch[:, :, band],
            (patch_size, patch_size),
            interpolation=cv2.INTER_CUBIC,
        )
    hr_patch = resized_patch
print("Resized HR Patch shape:", hr_patch.shape)


# 4. Generate a Low-Resolution Version of the Patch using bicubic interpolation
def generate_lr_patch(hr_patch, scale_factor):
    h, w, c = hr_patch.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    lr_patch = cv2.resize(hr_patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Upscale back to HR size (64x64)
    lr_patch_upscaled = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr_patch_upscaled


scale_factor = 2
lr_patch = generate_lr_patch(hr_patch, scale_factor)

# 5. Prepare the Input for the Model
# Model expects shape: (batch_size, height, width, channels, 1)
lr_patch_input = np.expand_dims(lr_patch, axis=0)  # Now: (1, 64, 64, 220)
lr_patch_input = np.expand_dims(lr_patch_input, axis=-1)  # Now: (1, 64, 64, 220, 1)

# 6. Enhance the Patch Using the Model
enhanced_patch = model.predict(lr_patch_input)
enhanced_patch = np.squeeze(enhanced_patch)  # Expected shape: (64, 64, 220)


# 7. Compute PSNR and SSIM
def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0  # images are normalized
    return 20 * math.log10(max_pixel / math.sqrt(mse))


psnr_value = compute_psnr(enhanced_patch, hr_patch)

# For SSIM, computed it on a representative spectral band, e.g., the middle band.
mid_band = hr_patch.shape[2] // 2
ssim_value = ssim(
    enhanced_patch[:, :, mid_band],
    hr_patch[:, :, mid_band],
    data_range=hr_patch[:, :, mid_band].max() - hr_patch[:, :, mid_band].min(),
)

print("PSNR between enhanced and HR patch:", psnr_value)
print("SSIM (for middle band) between enhanced and HR patch:", ssim_value)

# 8. Ploting the Results
plt.figure(figsize=(15, 5))

# Display the LR patch for a representative band
plt.subplot(1, 3, 1)
plt.imshow(lr_patch[:, :, mid_band], cmap="gray")
plt.title("LR Patch (Bicubic)")
plt.axis("off")

# Display the HR ground truth patch for the representative band
plt.subplot(1, 3, 2)
plt.imshow(hr_patch[:, :, mid_band], cmap="gray")
plt.title("HR Ground Truth")
plt.axis("off")

# Display the Enhanced patch for the representative band
plt.subplot(1, 3, 3)
plt.imshow(enhanced_patch[:, :, mid_band], cmap="gray")
plt.title(f"Enhanced Patch\nPSNR: {psnr_value:.2f} dB, SSIM: {ssim_value:.3f}")
plt.axis("off")

plt.tight_layout()
plt.show()
