# 🌱 AgriAI Plant Disease Detection

AgriAI is an AI-assisted plant disease classification system.

## Model

- Architecture: ResNet18
- Framework: PyTorch
- Dataset: PlantVillage
- Classes: 15
- Input: 224 × 224 RGB image

## Evaluation

Held-out PlantVillage test set:

- Accuracy: 99.84%
- Macro F1: 99.86%

These metrics represent performance on the held-out PlantVillage
test set and should not be interpreted as real-world field accuracy.

## Limitations

Real agricultural images can differ substantially from PlantVillage
images because of lighting, backgrounds, camera quality, disease
severity, plant varieties, and other environmental factors.

AgriAI should therefore be treated as an AI-assisted classification
tool rather than a definitive agricultural diagnosis.
