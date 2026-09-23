import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# AgriAI Vision v001
# ONNX Runtime Deployment
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "plant_disease_model.onnx"
CLASSES_PATH = BASE_DIR / "model" / "classes.json"


# ============================================================
# Load classes
# ============================================================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes_data = json.load(f)


if isinstance(classes_data, list):

    CLASS_NAMES = classes_data

elif isinstance(classes_data, dict) and "classes" in classes_data:

    CLASS_NAMES = classes_data["classes"]

elif isinstance(classes_data, dict) and "class_names" in classes_data:

    CLASS_NAMES = classes_data["class_names"]

elif isinstance(classes_data, dict) and "class_to_idx" in classes_data:

    CLASS_NAMES = [
        name
        for name, index in sorted(
            classes_data["class_to_idx"].items(),
            key=lambda x: x[1]
        )
    ]

else:

    raise ValueError(
        "Unsupported classes.json format."
    )


print("=" * 60)
print("AgriAI Vision v001")
print("=" * 60)
print("Classes:", len(CLASS_NAMES))


# ============================================================
# Load ONNX model
# ============================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = session.get_inputs()[0].name

print("Model:", MODEL_PATH)
print("Runtime: ONNX Runtime CPU")
print("Model loaded successfully.")


# ============================================================
# Readable class names
# ============================================================

def readable_label(label):

    replacements = {

        "Pepper__bell___Bacterial_spot":
            "Pepper Bell - Bacterial Spot",

        "Pepper__bell___healthy":
            "Pepper Bell - Healthy",

        "Potato___Early_blight":
            "Potato - Early Blight",

        "Potato___Late_blight":
            "Potato - Late Blight",

        "Potato___healthy":
            "Potato - Healthy",

        "Tomato_Bacterial_spot":
            "Tomato - Bacterial Spot",

        "Tomato_Early_blight":
            "Tomato - Early Blight",

        "Tomato_Late_blight":
            "Tomato - Late Blight",

        "Tomato_Leaf_Mold":
            "Tomato - Leaf Mold",

        "Tomato_Septoria_leaf_spot":
            "Tomato - Septoria Leaf Spot",

        "Tomato_Spider_mites_Two_spotted_spider_mite":
            "Tomato - Spider Mites",

        "Tomato__Target_Spot":
            "Tomato - Target Spot",

        "Tomato__Tomato_YellowLeaf__Curl_Virus":
            "Tomato - Yellow Leaf Curl Virus",

        "Tomato__Tomato_mosaic_virus":
            "Tomato - Mosaic Virus",

        "Tomato_healthy":
            "Tomato - Healthy",
    }

    return replacements.get(
        label,
        label.replace("_", " ")
    )


# ============================================================
# Image preprocessing
# Exact preprocessing used by the trained model
# ============================================================

def preprocess(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    image_array = (
        image_array - mean
    ) / std

    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    ).astype(np.float32)

    return image_array


# ============================================================
# Prediction
# ============================================================

def predict(image):

    if image is None:

        return (
            "## Please upload a plant leaf image.",
            {}
        )

    input_tensor = preprocess(image)

    outputs = session.run(
        None,
        {
            INPUT_NAME: input_tensor
        }
    )

    logits = outputs[0][0]

    # Stable softmax
    logits = (
        logits -
        np.max(logits)
    )

    probabilities = (
        np.exp(logits) /
        np.sum(np.exp(logits))
    )

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]


    results = {}

    for index in top_indices:

        results[
            readable_label(
                CLASS_NAMES[index]
            )
        ] = float(
            probabilities[index]
        )


    best_index = top_indices[0]

    best_probability = float(
        probabilities[best_index]
    )

    best_class = readable_label(
        CLASS_NAMES[best_index]
    )


    if best_probability >= 0.60:

        status = "HIGH CONFIDENCE"

    elif best_probability >= 0.40:

        status = "MODERATE CONFIDENCE"

    else:

        status = "LOW CONFIDENCE"


    report = f"""
# 🌱 AgriAI Analysis

## Prediction

**{best_class}**

## Confidence

**{best_probability * 100:.2f}%**

## Status

**{status}**

## Top 3 Predictions

"""


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        label = readable_label(
            CLASS_NAMES[index]
        )

        confidence = (
            probabilities[index] * 100
        )

        report += (
            f"{rank}. **{label}** "
            f"• {confidence:.2f}%\n"
        )


    report += """

---

### ⚠️ Important

This model was trained on the **PlantVillage dataset**.

PlantVillage images are relatively controlled compared with
photographs taken in real agricultural environments.

Performance can differ because of:

- Lighting
- Background
- Camera quality
- Leaf orientation
- Multiple leaves
- Multiple diseases
- Disease severity
- Dust and environmental damage
- Different plant varieties

This system is an **AI-assisted classification tool**,
not a definitive agricultural diagnosis.

### 🧠 Model

**Architecture:** ResNet18  
**Deployment format:** ONNX  
**Dataset:** PlantVillage  
**Classes:** 15  
**Input:** 224 × 224 RGB image  

**Held-out PlantVillage accuracy:** 99.84%  
**Held-out macro F1:** 99.86%

These metrics are from the held-out PlantVillage test set
and should not be interpreted as real-world field accuracy.
"""

    return report, results


# ============================================================
# Gradio interface
# ============================================================

with gr.Blocks(
    title="AgriAI Plant Disease Detection"
) as demo:

    gr.Markdown(
        """
# 🌱 AgriAI Plant Disease Detection

### AI-powered plant disease classification

Upload a plant leaf photograph or use your webcam.

AgriAI uses a trained **ResNet18 v001** model converted
to ONNX for lightweight CPU inference.
"""
    )


    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Plant Leaf",
                sources=[
                    "upload",
                    "webcam"
                ]
            )

            analyze_button = gr.Button(
                "🔬 Analyze Leaf",
                variant="primary"
            )


        with gr.Column():

            report_output = gr.Markdown(
                "Upload an image and click **Analyze Leaf**."
            )

            predictions_output = gr.Label(
                num_top_classes=3,
                label="Top Predictions"
            )


    analyze_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            report_output,
            predictions_output
        ]
    )


    gr.Markdown(
        """
---

### AgriAI Project

AI-assisted agricultural computer vision system.

Current vision model:

**ResNet18 v001 • 15 PlantVillage classes • ONNX Runtime**
"""
    )


# ============================================================
# Render server
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )
