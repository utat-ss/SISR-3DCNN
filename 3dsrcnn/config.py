from pathlib import Path

import tensorflow as tf

# Folders
DATA_FOLDER = './3dsrcnn/data/'
MODEL_FOLDER = './3dsrcnn/models/'

# File paths
MODEL_PATH = Path(MODEL_FOLDER, '3d_srcnn_model.keras')
CHECKPOINT_PATH = Path(MODEL_FOLDER, 'latest_checkpoint.keras')
NEXT_EPOCH_PATH = Path(MODEL_FOLDER, 'next_epoch.txt')

# 3D SRCNN Hyperparameters
learning_rate = 1e-5
HYPERPARAMETERS = {
    'epochs': 250,
    'batch_size': 2,
    'shuffle': True,
    'learning_rate': learning_rate,
    'optimizer': tf.keras.optimizers.Adam(learning_rate=learning_rate),
    'loss_function': 'mse'
}

# Patch parameters
PATCH_BANDS = 220
PATCH_SIZE = 64
STRIDE = 32

# Low-resolution patch parameters
SCALE_FACTOR = 2
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 1.0