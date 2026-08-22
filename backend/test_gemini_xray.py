import os
from google import genai
from google.genai import types


API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set.")


client = genai.Client(api_key=API_KEY)


IMAGE_PATH = r"E:\xray_project\nn.jpeg"


with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()


prompt = """
You are an AI assistant performing a structured second-opinion analysis of a
CHEST X-RAY.

Your task is to classify the image into ONE of these categories:

1. NORMAL
2. PNEUMONIA
3. COVID-19
4. UNCERTAIN

IMPORTANT:
- Analyze the IMAGE itself, not assumptions about the patient.
- First verify that the image is actually a chest X-ray.
- Examine both lungs, lung fields, hilar regions, costophrenic angles,
  cardiac silhouette, and visible pleural spaces.
- Consider the overall distribution, location, and appearance of pulmonary
  opacities.
- Look for findings such as focal consolidation, diffuse/bilateral opacities,
  infiltrates, ground-glass-type changes, or other visible abnormalities.
- Consider whether image quality, positioning, cropping, artifacts, or
  overlying objects interfere with interpretation.
- Do NOT invent findings that are not visible.
- Do NOT use patient history or information that is not present in the image.
- Do NOT automatically classify every abnormal X-ray as pneumonia.
- Do NOT automatically classify every bilateral opacity as COVID-19.
- COVID-19 should only be selected when the visible pattern is reasonably
  compatible with COVID-19; otherwise prefer PNEUMONIA, NORMAL, or UNCERTAIN.
- NORMAL should only be selected when there is no convincing radiographic
  abnormality relevant to the requested classification.
- If the image is not a usable chest X-ray or the findings cannot be
  reasonably distinguished, return UNCERTAIN.

CLASSIFICATION GUIDANCE:

NORMAL:
No convincing focal or diffuse pulmonary abnormality is visible.

PNEUMONIA:
Findings such as focal or multifocal air-space opacity, consolidation,
infiltrates, or other pulmonary abnormalities that are more compatible with
pneumonia.

COVID-19:
A pattern of pulmonary abnormalities that is reasonably compatible with
COVID-19, particularly bilateral or peripheral/multifocal involvement.
Do not select COVID-19 solely because abnormalities are bilateral.

UNCERTAIN:
The image is inadequate, heavily obscured, substantially affected by
artifacts, or the visible findings cannot be reasonably distinguished between
the categories.

CONFIDENCE:
Use HIGH only when the image provides reasonably clear evidence for the
selected category.
Use MEDIUM when the findings support the category but meaningful uncertainty
remains.
Use LOW when the classification is weak or the image has limitations.
UNCERTAIN should normally have LOW confidence.

OUTPUT:
Return ONLY valid JSON.
Do not include markdown.
Do not include ```json.
Do not include additional text.

Use exactly this structure:

{
  "is_chest_xray": true,
  "prediction": "NORMAL",
  "confidence": "HIGH",
  "reason": "Short explanation of the visible findings supporting the classification."
}

Allowed values:

"is_chest_xray":
true or false

"prediction":
"NORMAL"
"PNEUMONIA"
"COVID-19"
"UNCERTAIN"

"confidence":
"HIGH"
"MEDIUM"
"LOW"

If is_chest_xray is false, prediction MUST be "UNCERTAIN"
and confidence MUST be "LOW".

Keep the reason concise and describe only visible image findings.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        ),
        prompt,
    ],
)


print("\n" + "=" * 60)
print("GEMINI X-RAY RESULT")
print("=" * 60)

print(response.text)

print("=" * 60)