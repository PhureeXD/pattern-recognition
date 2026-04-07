# Project: IEEE-CIS Fraud Detection (6-Person ML vs DL Project)

## 1. Project Overview

We are building a machine learning pipeline to predict fraudulent transactions using the IEEE-CIS Fraud Detection dataset. The core objective is to compare Deep Learning architectures (TabNet, AutoEncoder) against advanced Tree-based Machine Learning (XGBoost, LightGBM) on tabular data, enhanced by Feature Synthesis and Stacking Ensembles.

## 2. Hardware Constraints & Memory Management (CRITICAL)

Our team operates on limited hardware: Google Colab T4 (12GB RAM, 16GB VRAM) and local NVIDIA RTX 3050 Ti (4GB VRAM).
**Strict Coding Rules for AI:**

- **NEVER** load the raw full CSV files into memory multiple times.
- **ALWAYS** use downcasting functions immediately after loading data (e.g., `float64` -> `float32`, `int64` -> `int8/int16`).
- **FORMAT:** Always convert and save intermediate cleaned datasets as `.parquet` files. Do not use `.csv` for intermediate steps.
- **LIBRARY:** Prefer `polars` over `pandas` for large aggregations. If `pandas` is used, utilize chunking or memory-efficient types.
- **TRAINING:** Use GPU acceleration for all models (e.g., `tree_method='gpu_hist'` for XGBoost, `device='cuda'` for PyTorch).

## 3. Team Division & Modules (6 Members)

When assisting a user, ask which module they are working on:

- **Team 1: Data Engineering (Members 1 & 2)**
  - _Member 1:_ `Identity` table and `TransactionDT` (Time-based features, Categorical encoding).
  - _Member 2:_ `Transaction` numericals, specifically `V1-V339` (PCA, dimensionality reduction, handling nulls).
- **Team 2: Core Modeling (Members 3, 4 & 5)**
  - _Member 3 (ML Baseline):_ XGBoost, LightGBM, CatBoost with Optuna tuning.
  - _Member 4 (DL TabNet):_ PyTorch `pytorch-tabnet`, attention masks, feature importance.
  - _Member 5 (DL AutoEncoder):_ Anomaly detection training ONLY on Class 0 (normal transactions), calculating reconstruction error.
- **Team 3: Evaluation & Ensembling (Member 6)**
  - _Member 6:_ Cross-validation strategy, Stacking Ensemble (combining predictions from Members 3, 4, 5), and Error Analysis (SHAP values, ROC-AUC comparison).

## 4. Project Milestones & Deliverables

- **Phase 1: Project Update**
  - **Deadline:** 16/4/24.
  - **Goal:** Have the first baseline models ready (both ML and DL).
  - **Required Content:** Problem setup, memory management strategy, early baseline results, and plan.
- **Phase 2: Final Project Presentation**
  - **Deadline:** XX/5/24.
  - **Required Content:** Complete pipeline, Metrics (ROC-AUC), Ensembling results.
  - **Error Analysis Rule (CRITICAL):** Must explicitly compare DL vs ML. Explain which method is better, what the model does well/not well, and why. Generate SHAP value visualizations and TabNet attention masks to support this.

## 5. Technology Stack

- **Data Processing:** `polars`, `pandas`, `numpy`, `scikit-learn` (PCA)
- **Machine Learning:** `xgboost`, `lightgbm`, `catboost`, `optuna`
- **Deep Learning:** `pytorch`, `pytorch-tabnet`
- **Evaluation:** `shap`, `matplotlib`, `seaborn`
