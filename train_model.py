"""
==============================================================
 XrayAI - Medical Chest X-Ray Training Pipeline
 TorchXRayVision 1.5.2
 DenseNet121 Medical Pretrained
==============================================================
"""

import os
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import torchxrayvision as xrv

warnings.filterwarnings("ignore")

###############################################################
# CONFIGURATION
###############################################################

class CFG:

    # Dataset
    DATASET_PATH = r"E:\xray_project\dataset"

    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VAL_DIR = os.path.join(DATASET_PATH, "val")
    TEST_DIR = os.path.join(DATASET_PATH, "test")

    # Image
    IMAGE_SIZE = 224

    # Training
    NUM_CLASSES = 3
    BATCH_SIZE = 8
    NUM_WORKERS = 0
    PIN_MEMORY = False
    USE_AMP = False

    EPOCHS = 30

    LR = 1e-4
    WEIGHT_DECAY = 1e-4

    RANDOM_SEED = 42

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    SAVE_DIR = "checkpoints"

os.makedirs(CFG.SAVE_DIR, exist_ok=True)

###############################################################
# RANDOM SEED
###############################################################

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.RANDOM_SEED)

###############################################################
# CLASS LABELS
###############################################################

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

###############################################################
# CUSTOM DATASET
###############################################################

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

            class_folder = self.root_dir / class_name

            if not class_folder.exists():
                continue

            for image_path in class_folder.iterdir():

                if image_path.suffix.lower() in valid_extensions:

                    self.samples.append(
                        (
                            str(image_path),
                            CLASS_TO_IDX[class_name]
                        )
                    )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        # Get image path and label
        image_path, label = self.samples[idx]

        # Load X-ray as grayscale
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        # Handle unreadable/corrupted image
        if image is None:

            print(
                f"Warning: Could not read image: {image_path}"
            )

            image = np.zeros(
                (
                    CFG.IMAGE_SIZE,
                    CFG.IMAGE_SIZE
                ),
                dtype=np.float32
            )

        # Convert to float32
        image = image.astype(
            np.float32
        )

        # Ensure contiguous NumPy array
        image = np.ascontiguousarray(
            image
        )

        # TorchXRayVision medical normalization
        image = xrv.datasets.normalize(
            image,
            255
        )

        # Add channel dimension
        # (H, W) -> (1, H, W)
        image = image[
            None,
            :,
            :
        ]

        # TorchXRayVision center crop
        image = xrv.datasets.XRayCenterCrop()(
            image
        )

        # Resize to configured size
        image = xrv.datasets.XRayResizer(
            CFG.IMAGE_SIZE
        )(
            image
        )

        # Convert NumPy array to PyTorch Tensor
        image = torch.from_numpy(
            image
        ).float()

        return image, label

###############################################################
# DATASETS
###############################################################

train_dataset = ChestXRayDataset(CFG.TRAIN_DIR)

val_dataset = ChestXRayDataset(CFG.VAL_DIR)

test_dataset = ChestXRayDataset(CFG.TEST_DIR)

###############################################################
# DATALOADERS
###############################################################

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY
)

###############################################################
# VERIFY DATASET
###############################################################

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Testing Images    : {len(test_dataset)}")

print("\nClasses")

for k, v in CLASS_TO_IDX.items():
    print(f"{v} -> {k}")

print("=" * 60)
print("Device :", CFG.DEVICE)
print("=" * 60)

sample_image, sample_label = train_dataset[0]

print("Sample Image Shape :", sample_image.shape)
print("Sample Label :", sample_label)


###############################################################
# CLASS WEIGHTS
###############################################################

train_labels = [label for _, label in train_dataset.samples]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(CFG.NUM_CLASSES),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(CFG.DEVICE)

print("\nClass Weights")
print(class_weights)

###############################################################
# MEDICAL PRETRAINED MODEL
###############################################################

print("\nLoading Medical DenseNet121...")

model = xrv.models.DenseNet(
    weights="densenet121-res224-all"
)

# Disable original 18-class operating thresholds
# because we are replacing the original classifier
# with our custom 3-class classifier.
model.op_threshs = None

print("Medical pretrained weights loaded.")

###############################################################
# REPLACE CLASSIFIER
###############################################################

in_features = model.classifier.in_features

model.classifier = nn.Sequential(

    nn.Linear(in_features,512),

    nn.ReLU(inplace=True),

    nn.Dropout(0.4),

    nn.Linear(512,CFG.NUM_CLASSES)

)

model = model.to(CFG.DEVICE)

print(model.classifier)


###############################################################
# FREEZE BACKBONE
###############################################################

for param in model.parameters():
    param.requires_grad=False

for param in model.classifier.parameters():
    param.requires_grad=True

print("\nBackbone Frozen")


###############################################################
# LOSS FUNCTION
###############################################################

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=0.1

)

###############################################################
# OPTIMIZER
###############################################################

optimizer = optim.AdamW(

    model.classifier.parameters(),

    lr=CFG.LR,

    weight_decay=CFG.WEIGHT_DECAY

)


###############################################################
# LEARNING RATE SCHEDULER
###############################################################

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2

)

###############################################################
# MODEL SUMMARY
###############################################################

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print("\nModel Summary")
print("-"*50)

print(f"Total Parameters     : {total:,}")
print(f"Trainable Parameters : {trainable:,}")


###############################################################
# TRAINING FUNCTION
###############################################################

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100*correct/total:.2f}%"
        )

    epoch_loss = running_loss / len(loader)

    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


###############################################################
# VALIDATION FUNCTION
###############################################################

def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    all_preds = []

    all_labels = []

    with torch.no_grad():

        progress = tqdm(loader, desc="Validation", leave=False)

        for images, labels in progress:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()

            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%"
            )

    epoch_loss = running_loss / len(loader)

    epoch_acc = 100 * correct / total

    return (
        epoch_loss,
        epoch_acc,
        np.array(all_labels),
        np.array(all_preds)
    )
     


###############################################################
# CHECKPOINT PATHS
###############################################################

CHECKPOINT_PATH = os.path.join(
    CFG.SAVE_DIR,
    "last_checkpoint.pth"
)

BEST_MODEL_PATH = os.path.join(
    CFG.SAVE_DIR,
    "best_model.pth"
)


###############################################################
# TRAINING STATE
###############################################################

best_val_acc = 0.0
start_epoch = 0
early_counter = 0

EARLY_STOPPING_PATIENCE = 5

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}


###############################################################
# CONFIGURE FINE-TUNING STAGE
###############################################################

def configure_training_stage(model, epoch):

    # ---------------------------------------------------------
    # First freeze everything
    # ---------------------------------------------------------

    for param in model.parameters():
        param.requires_grad = False


    # =========================================================
    # STAGE 1
    # Epochs 1 - 5
    # Train classifier only
    # =========================================================

    if epoch < 5:

        stage = 1
        learning_rate = 1e-4

        for param in model.classifier.parameters():
            param.requires_grad = True

        print("\nFine-Tuning Stage 1")
        print("Training: Classifier Only")
        print(f"Learning Rate: {learning_rate}")


    # =========================================================
    # STAGE 2
    # Epochs 6 - 15
    # Train DenseBlock4 + norm5 + classifier
    # =========================================================

    elif epoch < 15:

        stage = 2
        learning_rate = 1e-5

        for name, param in model.named_parameters():

            if (
                "denseblock4" in name
                or "norm5" in name
                or "classifier" in name
            ):
                param.requires_grad = True

        print("\nFine-Tuning Stage 2")
        print("Training: DenseBlock4 + norm5 + Classifier")
        print(f"Learning Rate: {learning_rate}")


    # =========================================================
    # STAGE 3
    # Epochs 16 - 30
    # Fine-tune entire network
    # =========================================================

    else:

        stage = 3
        learning_rate = 1e-6

        for param in model.parameters():
            param.requires_grad = True

        print("\nFine-Tuning Stage 3")
        print("Training: Entire Medical DenseNet121")
        print(f"Learning Rate: {learning_rate}")


    return stage, learning_rate


###############################################################
# CHECK FOR EXISTING CHECKPOINT
###############################################################

checkpoint = None

if os.path.exists(CHECKPOINT_PATH):

    print("\nCheckpoint Found!")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=CFG.DEVICE
    )

    # Load model weights
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Next epoch to train
    start_epoch = checkpoint["epoch"] + 1

    # Restore best accuracy
    best_val_acc = checkpoint.get(
        "best_accuracy",
        0.0
    )

    # Restore training history
    history = checkpoint.get(
        "history",
        history
    )

    # Restore early stopping counter
    early_counter = checkpoint.get(
        "early_counter",
        0
    )

    print(
        f"Checkpoint loaded successfully."
    )

    print(
        f"Resuming from Epoch {start_epoch + 1}"
    )

else:

    print("\nNo checkpoint found.")
    print("Starting fresh training...")


###############################################################
# CONFIGURE CORRECT STAGE FOR STARTING / RESUMING
###############################################################

current_stage, current_lr = configure_training_stage(
    model,
    start_epoch
)


###############################################################
# CREATE OPTIMIZER FOR CURRENT STAGE
###############################################################

optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=current_lr,

    weight_decay=CFG.WEIGHT_DECAY
)


###############################################################
# CREATE SCHEDULER FOR CURRENT OPTIMIZER
###############################################################

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)


###############################################################
# RESTORE OPTIMIZER AND SCHEDULER
###############################################################

if checkpoint is not None:

    saved_stage = checkpoint.get(
        "training_stage",
        current_stage
    )

    # Only restore optimizer state when checkpoint
    # belongs to the same fine-tuning stage.
    if saved_stage == current_stage:

        try:

            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

            print(
                "Optimizer and scheduler restored."
            )

        except (ValueError, KeyError) as error:

            print(
                "Optimizer state could not be restored."
            )

            print(
                "Using fresh optimizer for current stage."
            )

            print(
                f"Reason: {error}"
            )

    else:

        print(
            "Fine-tuning stage changed."
        )

        print(
            "Using fresh optimizer and scheduler."
        )


###############################################################
# DISPLAY TRAINABLE PARAMETERS
###############################################################

trainable_parameters = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)

print("\n" + "=" * 60)

print(
    f"Starting Fine-Tuning Stage : {current_stage}"
)

print(
    f"Trainable Parameters       : "
    f"{trainable_parameters:,}"
)

print(
    f"Starting Epoch             : "
    f"{start_epoch + 1}"
)

print("=" * 60)


###############################################################
# START TRAINING
###############################################################

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(
    start_epoch,
    CFG.EPOCHS
):

    print(
        f"\nEpoch [{epoch + 1}/{CFG.EPOCHS}]"
    )


    ###########################################################
    # DETERMINE REQUIRED TRAINING STAGE
    ###########################################################

    required_stage, required_lr = (
        configure_training_stage(
            model,
            epoch
        )
    )


    ###########################################################
    # IF TRAINING STAGE CHANGES
    ###########################################################

    if required_stage != current_stage:

        print("\n" + "=" * 60)

        print(
            f"Changing Fine-Tuning Stage: "
            f"{current_stage} -> {required_stage}"
        )

        print("=" * 60)

        current_stage = required_stage
        current_lr = required_lr


        #######################################################
        # CREATE NEW OPTIMIZER
        #######################################################

        optimizer = optim.AdamW(

            filter(
                lambda p: p.requires_grad,
                model.parameters()
            ),

            lr=current_lr,

            weight_decay=CFG.WEIGHT_DECAY
        )


        #######################################################
        # CREATE NEW SCHEDULER
        #######################################################

        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(

                optimizer,

                mode="max",

                factor=0.5,

                patience=2

            )
        )


        #######################################################
        # DISPLAY NEW TRAINABLE PARAMETER COUNT
        #######################################################

        trainable_parameters = sum(

            p.numel()

            for p in model.parameters()

            if p.requires_grad
        )

        print(
            f"Trainable Parameters: "
            f"{trainable_parameters:,}"
        )


    ###########################################################
    # TRAIN
    ###########################################################

    train_loss, train_acc = train_one_epoch(

        model,

        train_loader,

        criterion,

        optimizer,

        CFG.DEVICE
    )


    ###########################################################
    # VALIDATE
    ###########################################################

    val_loss, val_acc, _, _ = validate(

        model,

        val_loader,

        criterion,

        CFG.DEVICE
    )


    ###########################################################
    # UPDATE SCHEDULER
    ###########################################################

    scheduler.step(
        val_acc
    )


    ###########################################################
    # SAVE HISTORY
    ###########################################################

    history["train_loss"].append(
        train_loss
    )

    history["train_acc"].append(
        train_acc
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_acc"].append(
        val_acc
    )


    ###########################################################
    # PRINT EPOCH RESULTS
    ###########################################################

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1} Results"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc:.2f}%"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc:.2f}%"
    )

    print(
        f"Stage      : "
        f"{current_stage}"
    )

    print(
        f"Learning Rate : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print("-" * 60)


    ###########################################################
    # SAVE BEST MODEL
    ###########################################################

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        early_counter = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            "Best Model Saved!"
        )

    else:

        early_counter += 1


    ###########################################################
    # SAVE CHECKPOINT AFTER EVERY EPOCH
    ###########################################################

    torch.save(

        {

            "epoch":
                epoch,

            "training_stage":
                current_stage,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "best_accuracy":
                best_val_acc,

            "early_counter":
                early_counter,

            "history":
                history

        },

        CHECKPOINT_PATH

    )


    print(
        f"Checkpoint saved after Epoch "
        f"{epoch + 1}"
    )


    ###########################################################
    # EARLY STOPPING
    ###########################################################

    ###############################################################
# EARLY STOPPING DISABLED
# Training will always continue until TOTAL_EPOCHS
###############################################################

# Early stopping intentionally disabled.
# best_model.pth will still preserve the model
# with the highest validation accuracy.

###############################################################
# TRAINING COMPLETE
###############################################################

print("\n" + "=" * 60)

print(
    "TRAINING FINISHED"
)

print(
    f"Best Validation Accuracy : "
    f"{best_val_acc:.2f}%"
)

print("=" * 60)


