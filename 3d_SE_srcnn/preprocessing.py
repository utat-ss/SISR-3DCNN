# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

# Get the root directory of the repository
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define input and output directories
RAW_DIR = PROJECT_DIR / "raw_data"
PROCESSED_DIR = PROJECT_DIR / "processed_data"

# Create output directory if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Raw data:", RAW_DIR)
print("Processed data:", PROCESSED_DIR)

"""Load pavia dataset"""

# ============================================================
# STEP 2 — LOAD ALL HYPERSPECTRAL DATACUBES
# ============================================================

def load_all_datacubes(raw_dir):
    """
    Loads all four .npy hyperspectral datasets.

    Each cube is expected to have shape:
        Height x Width x Spectral Bands
    """

    dataset_files = [
        "cuprite512.npy",
        "ksc512.npy",
        "pavia512.npy",
        "indian_pine_array.npy"
    ]

    cubes = {}

    for filename in dataset_files:

        path = raw_dir / filename

        # Load .npy file
        cube = np.load(path)

        # Make sure it is a 3D hyperspectral cube
        if cube.ndim != 3:
            raise ValueError(
                f"{filename}: expected a 3D cube, "
                f"but got {cube.shape}"
            )

        # Store cube using dataset name
        dataset_name = path.stem
        cubes[dataset_name] = cube

        print(f"\nLoaded: {dataset_name}")
        print(f"Shape: {cube.shape}")
        print(f"dtype: {cube.dtype}")
        print(f"Range: {cube.min()} to {cube.max()}")

    return cubes

# Load all four datasets
cubes = load_all_datacubes(RAW_DIR)

# ============================================================
# STEP 3 — EXTRACT 64 x 64 HYPERSPECTRAL PATCHES
# ============================================================

def extract_patches(
    cube,
    patch_size=64,
    stride=64
):
    """
    Splits one HSI cube spatially into 64 x 64 patches
    while preserving ALL spectral bands.

    Input:
        H x W x Bands

    Output:
        N x 64 x 64 x Bands
    """

    height, width, bands = cube.shape

    patches = []

    for row in range(
        0,
        height - patch_size + 1,
        stride
    ):
        for col in range(
            0,
            width - patch_size + 1,
            stride
        ):

            patch = cube[
                row:row + patch_size,
                col:col + patch_size,
                :
            ]

            patches.append(patch)

    if len(patches) == 0:
        raise ValueError(
            f"Cube {cube.shape} is too small for "
            f"{patch_size}x{patch_size} patches."
        )

    return np.stack(patches)

# ============================================================
# STEP 3B — EXTRACT PATCHES FROM ALL FOUR DATASETS
# ============================================================

all_hr_patches = {}

for dataset_name, cube in cubes.items():

    patches = extract_patches(
        cube,
        patch_size=64,
        stride=64
    )

    all_hr_patches[dataset_name] = patches

    print(f"\n{dataset_name}")
    print(f"Original cube: {cube.shape}")
    print(f"HR patches:    {patches.shape}")

# ============================================================
# STEP 4A — DEFINE LOW-RESOLUTION HSI GENERATION
# ============================================================

from scipy.ndimage import gaussian_filter, zoom


def generate_low_res_patch(
    hr_patch,
    scale_factor,
    sigma=1.0
):
    """
    Creates a low-resolution HSI patch from an HR patch.

    Process:
        1. Gaussian blur
        2. Nearest-neighbor spatial downsampling

    Input shape:
        64 x 64 x Bands

    Output:
        x2 -> 32 x 32 x Bands
        x4 -> 16 x 16 x Bands
    """

    # --------------------------------------------------------
    # STEP 4A.1 — Gaussian blur
    # --------------------------------------------------------

    blurred = gaussian_filter(
        hr_patch,
        sigma=(sigma, sigma, 0)
    )

    # sigma for:
    # height  = sigma
    # width   = sigma
    # bands   = 0
    #
    # Therefore spectral bands are not blurred together.


    # --------------------------------------------------------
    # STEP 4A.2 — Nearest-neighbor downsampling
    # --------------------------------------------------------

    lr_patch = zoom(
        blurred,
        zoom=(
            1 / scale_factor,
            1 / scale_factor,
            1
        ),
        order=0
    )

    # order=0 means nearest-neighbor.
    #
    # The final 1 means:
    # DON'T resize the spectral dimension.

    return lr_patch

# ============================================================
# STEP 4B — GENERATE x2 AND x4 LR PATCHES FOR ALL DATASETS
# ============================================================

all_lr_patches = {}


for dataset_name, hr_patches in all_hr_patches.items():

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    # Dictionary for this dataset
    all_lr_patches[dataset_name] = {}


    # --------------------------------------------------------
    # Generate x2 and x4
    # --------------------------------------------------------

    for scale_factor in [2, 4]:

        lr_patches = []

        # Process every HR patch
        for hr_patch in hr_patches:

            lr_patch = generate_low_res_patch(
                hr_patch,
                scale_factor=scale_factor
            )

            lr_patches.append(
                lr_patch
            )


        # Convert list -> NumPy array
        lr_patches = np.stack(
            lr_patches
        )


        # Store it
        all_lr_patches[
            dataset_name
        ][
            scale_factor
        ] = lr_patches


        print(
            f"x{scale_factor} LR patches:",
            lr_patches.shape
        )

# ============================================================
# STEP 5A — DEFINE BICUBIC UPSAMPLING
# ============================================================

def upsample_low_res_patch(
    lr_patch,
    scale_factor
):
    """
    Upsamples an LR hyperspectral patch back to 64 x 64.

    Input:
        x2 -> 32 x 32 x Bands
        x4 -> 16 x 16 x Bands

    Output:
        64 x 64 x Bands

    Spectral bands are NOT resized.
    """

    upsampled_patch = zoom(
        lr_patch,
        zoom=(
            scale_factor,
            scale_factor,
            1
        ),
        order=3
    )

    # order=3 -> cubic interpolation
    #
    # scale_factor affects:
    # height -> x2 or x4
    # width  -> x2 or x4
    # bands  -> x1 (unchanged)

    return upsampled_patch

# ============================================================
# STEP 5B — UPSAMPLE ALL x2 AND x4 LR DATASETS
# ============================================================

all_upsampled_patches = {}


for dataset_name, scale_data in all_lr_patches.items():

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    all_upsampled_patches[dataset_name] = {}


    # --------------------------------------------------------
    # Process both x2 and x4
    # --------------------------------------------------------

    for scale_factor in [2, 4]:

        lr_patches = scale_data[
            scale_factor
        ]

        upsampled_patches = []


        # ----------------------------------------------------
        # Upsample every LR patch
        # ----------------------------------------------------

        for lr_patch in lr_patches:

            upsampled_patch = (
                upsample_low_res_patch(
                    lr_patch,
                    scale_factor
                )
            )

            upsampled_patches.append(
                upsampled_patch
            )


        # Convert list -> NumPy array
        upsampled_patches = np.stack(
            upsampled_patches
        )


        # Store result
        all_upsampled_patches[
            dataset_name
        ][
            scale_factor
        ] = upsampled_patches


        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"x{scale_factor} LR:       ",
            lr_patches.shape
        )

        print(
            f"x{scale_factor} upsampled:",
            upsampled_patches.shape
        )

# ============================================================
# STEP 6A — ADD CHANNEL DIMENSION FOR 3D MODEL
# ============================================================

def add_channel_dimension(patches):
    """
    Adds the channel dimension required by the 3D model.

    Before:
        N x 64 x 64 x Bands

    After:
        N x 64 x 64 x Bands x 1
    """

    model_ready = patches[..., np.newaxis]

    return model_ready

# ============================================================
# STEP 6B — FORMAT ALL DATASETS FOR THE 3D-SE-SRCNN
# ============================================================

all_model_inputs = {}
all_model_targets = {}


for dataset_name in all_hr_patches.keys():

    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    all_model_inputs[dataset_name] = {}
    all_model_targets[dataset_name] = {}


    # --------------------------------------------------------
    # Process both x2 and x4
    # --------------------------------------------------------

    for scale_factor in [2, 4]:

        # ----------------------------------------
        # X = upsampled LR input
        # ----------------------------------------

        upsampled_patches = (
            all_upsampled_patches[
                dataset_name
            ][
                scale_factor
            ]
        )

        X = add_channel_dimension(
            upsampled_patches
        )


        # ----------------------------------------
        # Y = original HR ground truth
        # ----------------------------------------

        hr_patches = all_hr_patches[
            dataset_name
        ]

        Y = add_channel_dimension(
            hr_patches
        )


        # ----------------------------------------
        # Store arrays
        # ----------------------------------------

        all_model_inputs[
            dataset_name
        ][
            scale_factor
        ] = X

        all_model_targets[
            dataset_name
        ][
            scale_factor
        ] = Y


        # ----------------------------------------
        # Print shapes
        # ----------------------------------------

        print(f"\nx{scale_factor}")

        print(
            "X model input:",
            X.shape
        )

        print(
            "Y ground truth:",
            Y.shape
        )

# ============================================================
# STEP 6C — VERIFY MODEL-READY DATA
# ============================================================

for dataset_name in all_model_inputs.keys():

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    for scale_factor in [2, 4]:

        X = all_model_inputs[
            dataset_name
        ][
            scale_factor
        ]

        Y = all_model_targets[
            dataset_name
        ][
            scale_factor
        ]

        # X and Y must have same shape
        assert X.shape == Y.shape

        # Must be 5D:
        # N x H x W x Bands x Channel
        assert X.ndim == 5
        assert Y.ndim == 5

        # Spatial patches must be 64 x 64
        assert X.shape[1] == 64
        assert X.shape[2] == 64

        # Last dimension must be 1
        assert X.shape[-1] == 1
        assert Y.shape[-1] == 1

        # Make sure there are no NaNs
        assert not np.isnan(X).any()
        assert not np.isnan(Y).any()

        # Make sure there are no infinities
        assert np.isfinite(X).all()
        assert np.isfinite(Y).all()

        print(
            f"x{scale_factor}: ✓ "
            f"X {X.shape} | Y {Y.shape}"
        )

# ============================================================
# STEP 7A — DEFINE FUNCTION TO SAVE PROCESSED DATA
# ============================================================

def save_processed_dataset(
    dataset_name,
    scale_factor,
    X,
    Y
):
    """
    Saves the model-ready input (X) and
    high-resolution ground truth (Y).

    X:
        Upsampled LR-HSI
        Shape = N x 64 x 64 x Bands x 1

    Y:
        Original HR-HSI
        Shape = N x 64 x 64 x Bands x 1
    """

    # Create output folder
    output_dir = (
        PROCESSED_DIR
        / dataset_name
        / f"x{scale_factor}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model input
    np.save(
        output_dir / "X_model_input.npy",
        X
    )

    # Save ground truth
    np.save(
        output_dir / "Y_ground_truth.npy",
        Y
    )

    print(
        f"✓ Saved {dataset_name} x{scale_factor}"
    )

# ============================================================
# STEP 7B — SAVE ALL FOUR DATASETS
# ============================================================

for dataset_name in all_model_inputs.keys():

    print("\n" + "=" * 60)
    print(f"Saving: {dataset_name}")
    print("=" * 60)

    for scale_factor in [2, 4]:

        X = all_model_inputs[
            dataset_name
        ][
            scale_factor
        ]

        Y = all_model_targets[
            dataset_name
        ][
            scale_factor
        ]

        save_processed_dataset(
            dataset_name=dataset_name,
            scale_factor=scale_factor,
            X=X,
            Y=Y
        )

# ============================================================
# STEP 8 — VERIFY SAVED FILES
# ============================================================

for dataset_name in all_model_inputs.keys():

    print("\n" + "=" * 60)
    print(f"Checking: {dataset_name}")
    print("=" * 60)

    for scale_factor in [2, 4]:

        folder = (
            PROCESSED_DIR
            / dataset_name
            / f"x{scale_factor}"
        )

        X_path = folder / "X_model_input.npy"
        Y_path = folder / "Y_ground_truth.npy"

        # mmap_mode avoids loading the whole file into RAM
        X = np.load(
            X_path,
            mmap_mode="r"
        )

        Y = np.load(
            Y_path,
            mmap_mode="r"
        )

        print(f"\nx{scale_factor}")
        print("X:", X.shape, X.dtype)
        print("Y:", Y.shape, Y.dtype)

        # Sanity checks
        assert X.shape == Y.shape
        assert X.ndim == 5
        assert Y.ndim == 5
        assert X.shape[1] == 64
        assert X.shape[2] == 64
        assert X.shape[-1] == 1
        assert Y.shape[-1] == 1

        print("✓ Saved files verified")