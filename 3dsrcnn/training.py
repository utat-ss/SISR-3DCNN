"""Python code for training a SRCNN model."""

# external
import cv2
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from srcnn import srcnn_3D_333  # Adjust import based on your chosen model

# Load the dataset
data = scipy.io.loadmat("Indian_pines.mat")
# Inspect the keys to find the hyperspectral data variable (e.g., 'indian_pines')
hs_data = data["indian_pines"]  # Adjust the key as needed

# Convert to float and normalize (e.g., scale values between 0 and 1)
hs_data = hs_data.astype(np.float32)
hs_data /= np.max(hs_data)


def extract_patches(image, patch_size, stride):
    patches = []
    H, W, _ = image.shape  # Assuming image is [height, width, bands]
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i : i + patch_size, j : j + patch_size, :]
            patches.append(patch)
    return np.array(patches)


patch_size = 64  # Example patch size
stride = 32  # Overlapping patches
patches = extract_patches(hs_data, patch_size, stride)


def generate_lr_patch(hr_patch, scale_factor):
    # Downsample the HR patch to create the LR patch
    h, w, c = hr_patch.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    lr_patch = cv2.resize(hr_patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # Optionally, upscale the LR patch back to HR size for a direct supervised comparison
    lr_patch_upscaled = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
    return lr_patch_upscaled


scale_factor = 2
lr_patches = np.array([generate_lr_patch(patch, scale_factor) for patch in patches])

# Expand dimensions to add the channel dimension (required for 3D models)
lr_patches = lr_patches[..., np.newaxis]
patches = patches[..., np.newaxis]

X_train, X_temp, Y_train, Y_temp = train_test_split(
    lr_patches, patches, test_size=0.3, random_state=42
)
X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp, test_size=0.5, random_state=42
)


IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = patch_size, patch_size, hs_data.shape[2]
model = srcnn_3D_333(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model.compile(optimizer="adam", loss="mse")

# Train the model
history = model.fit(
    X_train, Y_train, batch_size=2, epochs=250, validation_data=(X_val, Y_val)
)

# Evaluate
test_loss = model.evaluate(X_test, Y_test)
print("Test Loss (MSE):", test_loss)

# Save the trained model to disk
model.save("srcnn_model.h5")

# Predict on a new LR patch (or full image patched and reassembled)
predicted_hr = model.predict(X_test[0:1])
