# Hallucination Detection with Unlabeled Data (Semi-Supervised)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)

### The Problem: Silent Failures in LLMs
Large Language Models (LLMs) often generate "fluent but false" information (Hallucinations). In enterprise RAG systems, these errors are dangerous.
**The Challenge:** Detecting these errors usually requires massive labeled datasets, which are expensive and slow to create.

### The Solution
We developed a **Semi-Supervised Pipeline** that detects hallucinations with high accuracy **without** needing a fully labeled training set. By using a "Teacher-Student" approach and an ensemble classifier, we successfully classify model errors even in low-resource settings.

---

## Architecture & Methodology

### 1. The "Teacher": Zero-Shot Pseudo-Labeling
We utilized **Mistral-7B** (via Zero-Shot Prompting) to act as a "Teacher."
*   We fed it unlabelled text and asked it to distinguish between factual statements and hallucinations.
*   This process generated **Pseudo-Labels**, effectively creating a synthetic dataset from scratch.

### 2. The "Student": Ensemble Classification
We used the synthetic data to train a robust ensemble model, combining the semantic understanding of Transformers with the statistical power of Gradient Boosting.

*   **Deep Learning Layer:** We fine-tuned **Multilingual DeBERTa-v3** to extract semantic embeddings from the text. DeBERTa was chosen for its superior performance on NLI (Natural Language Inference) tasks compared to BERT/RoBERTa.
*   **Boosting Layer:** We fed these embeddings into **CatBoost**, which learned to classify the final output based on the feature vectors.

---

## Key Results

*   **Accuracy:** The final ensemble model achieved an accuracy of **78%**, significantly outperforming the baseline.
*   **Data Efficiency:** Proved that robust safety monitors can be trained using **unlabeled data** by leveraging larger LLMs as annotators.
*   **Performance:** The combination of DeBERTa (Semantic) and CatBoost (Decision Trees) captured edge cases that single architectures missed.

---

## Tech Stack
*   **LLMs:** `Mistral-7B` (Pseudo-Labeling), `mDeBERTa-v3` (Feature Extraction)
*   **Classifiers:** `CatBoost` (Gradient Boosting)
*   **Libraries:** `PyTorch`, `Transformers`, `Scikit-Learn`
*   **Technique:** `Semi-Supervised Learning`, `Ensemble Learning`, `Zero-Shot Prompting`

---
