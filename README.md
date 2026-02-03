# Language Matters: Target-Language Supervision for Political Bias Detection in Turkish News

This repository contains the **model fine-tuning and evaluation code** for the paper:

**Language Matters: Target-Language Supervision for Political Bias Detection in Turkish News**  
Umut Ozbagriacik, Haim Dubossarsky  
(SIGTURK / EACL Workshop)

---

## Overview

This codebase supports the experimental results reported in the paper, focusing on:

- Transformer-based political bias classification in **Turkish news**
- Comparison between **monolingual (BERTurk)**, **multilingual (XLM-R, mBERT)**, and **decoder-based (Mistral-LoRA)** models
- **Fold-level cross-validation**, **fold-level ensembling**, and **cross-model ensembling**
- **Cross-lingual transfer experiments** (English → Turkish)

**Note:**  
Due to copyright restrictions, **news articles are not included** in this repository.  
The code assumes preprocessed datasets with text and labels.

---

## Models

The following models are supported:

- **BERTurk** (`dbmdz/bert-base-turkish-uncased`)
- **XLM-RoBERTa** (`xlm-roberta-base`)
- **mBERT** (`bert-base-multilingual-cased`)
- **Mistral-7B (LoRA fine-tuning)**  
  (`malhajar/Mistral-7B-Instruct-v0.2-turkish`)

All encoder models are fine-tuned as **3-class sequence classifiers**  
(left / centre / right).

---

## Training & Validation Strategy

- **Stratified 5-fold cross-validation** is used **for every model**
- Each fold produces a separate checkpoint
- For test-time evaluation:
  - Logits are averaged **across the 5 folds** (fold-level ensembling)
  - Final ensemble averages logits **across model architectures**

This reduces variance and improves robustness, especially in low-resource settings.

---

## Cross-Lingual Experiments

The repository also includes experiments where models are:

- Fine-tuned on **English political bias data**
- Directly evaluated on **Turkish test data** (zero-shot transfer)

These experiments demonstrate that **English supervision does not transfer effectively** to Turkish political bias detection.

---

## Evaluation

Reported metrics:

- Accuracy
- Precision
- Recall
- Macro-F1

Confusion matrices are generated to analyse class-wise behaviour  
(centre vs. partisan, left–right confusions).

---
