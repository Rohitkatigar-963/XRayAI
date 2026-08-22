import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


MODEL_NAME = "google/medsiglip-448"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("MEDSIGLIP TEST")
print("=" * 60)
print("Device:", DEVICE)

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading MedSigLIP model...")
model = AutoModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("MedSigLIP loaded successfully!")


# ============================================================
# CHANGE THIS TO ONE GOOGLE X-RAY THAT YOUR MODEL MISCLASSIFIES
# ============================================================

IMAGE_PATH = r"E:\xray_project\YOUR_IMAGE.jpg"


image = Image.open(IMAGE_PATH).convert("RGB")


# ============================================================
# CANDIDATE CLASSES
# ============================================================

labels = [
    "a chest X-ray showing COVID-19",
    "a chest X-ray showing pneumonia",
    "a normal chest X-ray"
]


print("\nRunning MedSigLIP inference...")


inputs = processor(
    text=labels,
    images=image,
    padding="max_length",
    return_tensors="pt"
).to(DEVICE)


with torch.no_grad():

    outputs = model(**inputs)

    logits = outputs.logits_per_image

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("MEDSIGLIP RESULTS")
print("=" * 60)

for label, probability in zip(
    labels,
    probabilities
):

    print(
        f"{label}: "
        f"{probability.item() * 100:.2f}%"
    )


best_index = torch.argmax(
    probabilities
).item()


print("\nMedSigLIP prediction:")
print(labels[best_index])

print("=" * 60)