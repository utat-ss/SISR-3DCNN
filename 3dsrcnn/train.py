import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
import preprocessing
from srcnn import srcnn_3D_333

# ========== Model-Related Code ==========
class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open(config.NEXT_EPOCH_PATH, 'w') as f:
            f.write(str(epoch + 1)) # Saves the next epoch to resume from

def does_checkpoint_exist():
    return os.path.exists(config.CHECKPOINT_PATH) and os.path.exists(config.NEXT_EPOCH_PATH)

def get_checkpoint_data():
    initial_epoch = 0
    model = tf.keras.models.load_model(
        config.CHECKPOINT_PATH,
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )

    with open(config.NEXT_EPOCH_PATH, 'r') as f:
        initial_epoch = int(f.read())

    return model, initial_epoch

# ========== Load Dataset ==========
data_folder = config.DATA_FOLDER
datacube_file_keys = {'indian_pines.mat': 'indian_pines', 'cuprite.mat': 'X'}
datacubes = []

for filename, data_key in datacube_file_keys.items():
    datacubes.append(preprocessing.load_datacube(data_folder, filename, data_key))

# ========== Extract Patches from Each Datacube ==========
patches = []
patch_size = config.PATCH_SIZE
stride = config.STRIDE

for datacube in datacubes:
    datacube_patches = preprocessing.extract_patches(datacube, patch_size, stride)
    patches.append(datacube_patches)

    print(f'Extracted {len(datacube_patches)} patches')
    
patches = np.concatenate(patches, axis=0)

# ========== Convert Each Patch into a Low-Resolution Datacube ==========
scale_factor = config.SCALE_FACTOR
low_res_patches = np.array([preprocessing.generate_low_res_patch(patch, scale_factor) for patch in patches])
upsampled_patches = np.array([preprocessing.upsample_low_res_patch(patch, scale_factor) for patch in low_res_patches])

# ========== Add Channel Dimension (Required for 3D Models) ==========
# Model expects shape: (batch_size, height, width, channels, 1)
patches = patches[..., np.newaxis]
upsampled_patches = upsampled_patches[..., np.newaxis]

# ========== Prepare Training, Validation, and Testing Pairs ==========
x_train, x_temp, y_train, y_temp = train_test_split(
    upsampled_patches, patches, test_size=0.3, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# ========== Set Up Model ==========
hyperparameters = config.HYPERPARAMETERS
initial_epoch = 0

if does_checkpoint_exist():
    model, initial_epoch = get_checkpoint_data()

else:
    img_height, img_width, img_channels = None, None, None      # Set as None for dynamic shapes
    model = srcnn_3D_333(img_height, img_width, img_channels)
    model.compile(optimizer=hyperparameters['optimizer'], loss=hyperparameters['loss_function'])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=config.CHECKPOINT_PATH,
    save_weights_only=False,
    save_best_only=False,
    save_freq='epoch',
    verbose=1
)

# ========== Train Model ==========
history = model.fit(
    x_train, y_train,
    batch_size=hyperparameters['batch_size'],
    epochs=hyperparameters['epochs'],
    initial_epoch=initial_epoch,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint_callback, EpochSaver()]
)

# ========== Save Trained Model ==========
model.save(config.MODEL_PATH)