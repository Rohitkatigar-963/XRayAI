"""
==============================================================
 XrayAI - Final Model Evaluation
 Medical Pretrained DenseNet121
 Tests best_model.pth on the untouched TEST dataset
==============================================================
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchxrayvision as xrv

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================================
# CONFIGURATION
# ============================================================

class CFG:

    DATASET_PATH = r"E:\xray_project\dataset"

    TEST_DIR = os.path.join(
        DATASET_PATH,
        "test"
    )

    MODEL_PATH = r"E:\xray_project\checkpoints\best_model.pth"

    IMAGE_SIZE = 224

    NUM_CLASSES = 3

    BATCH_SIZE = 8

    NUM_WORKERS = 0

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# CLASS LABELS
# MUST MATCH TRAINING EXACTLY
# ============================================================

CLASS_TO_IDX = {
    "COVID": 0,
    "NORMAL": 1,
    "PNEUMONIA": 2
}

IDX_TO_CLASS = {
    0: "COVID",
    1: "NORMAL",
    2: "PNEUMONIA"
}

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]


# ============================================================
# TEST DATASET
# Uses same preprocessing as train_model.py
# ============================================================

class ChestXRayDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.samples = []

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff"
        )

        for class_name in CLASS_TO_IDX.keys():

            class_folder = (
                self.root_dir /
                class_name
            )

            if not class_folder.exists():

                print(
                    f"WARNING: Folder not found: "
                    f"{class_folder}"
                )

                continue

            for image_path in class_folder.iterdir():

                if (
                    image_path.suffix.lower()
                    in valid_extensions
                ):

                    self.samples.append(
                        (
                            str(image_path),
                            CLASS_TO_IDX[
                                class_name
                            ]
                        )
                    )

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):

        image_path, label = (
            self.samples[idx]
        )

        # Load as grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise ValueError(
                f"Could not read image: "
                f"{image_path}"
            )

        # Convert to float32
        image = image.astype(
            np.float32
        )

        image = np.ascontiguousarray(
            image
        )

        # TorchXRayVision normalization
        image = xrv.datasets.normalize(
            image,
            255
        )

        # Add channel dimension
        # H,W -> 1,H,W
        image = image[
            None,
            :,
            :
        ]

        # Center crop
        image = (
            xrv.datasets
            .XRayCenterCrop()(
                image
            )
        )

        # Resize to 224x224
        image = (
            xrv.datasets
            .XRayResizer(
                CFG.IMAGE_SIZE
            )(
                image
            )
        )

        # Convert to tensor
        image = torch.from_numpy(
            image
        ).float()

        return (
            image,
            label,
            image_path
        )


# ============================================================
# VERIFY MODEL FILE
# ============================================================

if not os.path.exists(
    CFG.MODEL_PATH
):

    raise FileNotFoundError(

        "\nBest model not found!\n"

        f"Expected location:\n"
        f"{CFG.MODEL_PATH}\n"
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "XRAYAI FINAL MODEL EVALUATION"
)

print(
    "=" * 60
)


test_dataset = ChestXRayDataset(
    CFG.TEST_DIR
)


if len(test_dataset) == 0:

    raise RuntimeError(
        "No test images were found."
    )


test_loader = DataLoader(

    test_dataset,

    batch_size=CFG.BATCH_SIZE,

    shuffle=False,

    num_workers=CFG.NUM_WORKERS
)


print(
    f"\nTest Images : "
    f"{len(test_dataset)}"
)

print(
    f"Device      : "
    f"{CFG.DEVICE}"
)


# ============================================================
# DISPLAY TEST CLASS COUNTS
# ============================================================

class_counts = {
    name: 0
    for name in CLASS_NAMES
}


for _, label in test_dataset.samples:

    class_name = IDX_TO_CLASS[
        label
    ]

    class_counts[
        class_name
    ] += 1


print(
    "\nTest Dataset Distribution"
)

print(
    "-" * 40
)


for class_name in CLASS_NAMES:

    print(
        f"{class_name:<12}: "
        f"{class_counts[class_name]}"
    )


# ============================================================
# LOAD MEDICAL PRETRAINED DENSENET121
# ============================================================

print(
    "\nLoading Medical DenseNet121..."
)


model = xrv.models.DenseNet(

    weights="densenet121-res224-all"

)


# Disable original thresholds
model.op_threshs = None


# ============================================================
# RECREATE EXACT SAME CLASSIFIER
# ============================================================

in_features = (
    model.classifier.in_features
)


model.classifier = nn.Sequential(

    nn.Linear(
        in_features,
        512
    ),

    nn.ReLU(
        inplace=True
    ),

    nn.Dropout(
        0.4
    ),

    nn.Linear(
        512,
        CFG.NUM_CLASSES
    )

)


# ============================================================
# LOAD BEST MODEL WEIGHTS
# ============================================================

print(
    "\nLoading best_model.pth..."
)


state_dict = torch.load(

    CFG.MODEL_PATH,

    map_location=CFG.DEVICE

)


model.load_state_dict(
    state_dict
)


model = model.to(
    CFG.DEVICE
)


model.eval()


print(
    "Best model loaded successfully."
)


# ============================================================
# RUN TEST EVALUATION
# ============================================================

print(
    "\nRunning evaluation on test dataset..."
)


all_labels = []

all_predictions = []

all_paths = []


with torch.no_grad():

    for (
        images,
        labels,
        paths
    ) in test_loader:

        images = images.to(
            CFG.DEVICE
        )

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )

        all_paths.extend(
            paths
        )


all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)


# ============================================================
# OVERALL TEST ACCURACY
# ============================================================

accuracy = accuracy_score(

    all_labels,

    all_predictions

)


print(
    "\n" +
    "=" * 60
)

print(
    "FINAL TEST RESULTS"
)

print(
    "=" * 60
)


print(

    f"\nOverall Test Accuracy : "
    f"{accuracy * 100:.2f}%"

)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report"
)

print(
    "-" * 60
)


report = classification_report(

    all_labels,

    all_predictions,

    labels=[
        0,
        1,
        2
    ],

    target_names=CLASS_NAMES,

    digits=4,

    zero_division=0

)


print(
    report
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions,

    labels=[
        0,
        1,
        2
    ]

)


print(
    "\nConfusion Matrix"
)

print(
    "Rows    = Actual Class"
)

print(
    "Columns = Predicted Class"
)

print()


header = (
    f"{'Actual':<15}"
    f"{'COVID':>12}"
    f"{'NORMAL':>12}"
    f"{'PNEUMONIA':>15}"
)

print(
    header
)

print(
    "-" * 54
)


for i, class_name in enumerate(
    CLASS_NAMES
):

    print(

        f"{class_name:<15}"

        f"{cm[i][0]:>12}"

        f"{cm[i][1]:>12}"

        f"{cm[i][2]:>15}"

    )


# ============================================================
# IMPORTANT MISCLASSIFICATION ANALYSIS
# ============================================================

covid_to_normal = cm[0][1]

covid_to_pneumonia = cm[0][2]

normal_to_covid = cm[1][0]

normal_to_pneumonia = cm[1][2]

pneumonia_to_covid = cm[2][0]

pneumonia_to_normal = cm[2][1]


print(
    "\n" +
    "=" * 60
)

print(
    "MISCLASSIFICATION ANALYSIS"
)

print(
    "=" * 60
)


print(

    f"\nCOVID predicted as NORMAL      : "
    f"{covid_to_normal}"

)

print(

    f"COVID predicted as PNEUMONIA   : "
    f"{covid_to_pneumonia}"

)

print(

    f"NORMAL predicted as COVID      : "
    f"{normal_to_covid}"

)

print(

    f"NORMAL predicted as PNEUMONIA  : "
    f"{normal_to_pneumonia}"

)

print(

    f"PNEUMONIA predicted as COVID   : "
    f"{pneumonia_to_covid}"

)

print(

    f"PNEUMONIA predicted as NORMAL  : "
    f"{pneumonia_to_normal}"

)


# ============================================================
# SAVE MISCLASSIFIED IMAGE PATHS
# ============================================================

output_file = (
    "misclassified_images.txt"
)


with open(

    output_file,

    "w",

    encoding="utf-8"

) as file:

    for (
        true_label,
        predicted_label,
        image_path
    ) in zip(

        all_labels,

        all_predictions,

        all_paths

    ):

        if (
            true_label
            != predicted_label
        ):

            file.write(

                f"IMAGE: {image_path}\n"

            )

            file.write(

                f"ACTUAL: "
                f"{IDX_TO_CLASS[true_label]}\n"

            )

            file.write(

                f"PREDICTED: "
                f"{IDX_TO_CLASS[predicted_label]}\n"

            )

            file.write(

                "-" * 60 +
                "\n"

            )


print(
    "\nMisclassified image details "
    "saved to:"
)

print(
    output_file
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "TESTING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)