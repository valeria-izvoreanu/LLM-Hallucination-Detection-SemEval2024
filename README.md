# Hallucination Detection with Unlabeled Data (Semi-Supervised)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)

### The Problem: Silent Failures in LLMs
Large Language Models (LLMs) often generate "fluent but false" information (Hallucinations). In enterprise RAG systems, these errors are dangerous.
**The Challenge:** Detecting these errors usually requires massive labeled datasets, which are expensive and slow to create.

### The Solution
We developed a **Semi-Supervised Pipeline** that detects hallucinations with high accuracy **without** needing a fully labeled training set. By using a "Teacher-Student" approach and an ensemble classifier, we successfully classify model errors even in low-resource settings.

---

## Architecture: Hybrid Ensemble

We combined synthetic and gold data streams to maximize performance.

1.  **Pseudo-Labeling (Mistral-7B):** Used Zero-Shot prompting to generate synthetic labels for unlabeled text.
2.  **DeBERTa-v3:** Fine-tuned on the **Synthetic Data** to learn general semantic patterns from high-volume, noisy input.
3.  **CatBoost:** Trained exclusively on **Gold (Validation) Data** to anchor predictions to human-verified ground truth.
4.  **Final Output:** A weighted ensemble of both models, achieving **78% Accuracy**.

---

## Tech Stack
*   **LLMs:** `Mistral-7B` (Pseudo-Labeling), `mDeBERTa-v3` (Feature Extraction)
*   **Classifiers:** `CatBoost` (Gradient Boosting)
*   **Libraries:** `PyTorch`, `Transformers`, `Scikit-Learn`
*   **Technique:** `Semi-Supervised Learning`, `Ensemble Learning`, `Zero-Shot Prompting`

---
