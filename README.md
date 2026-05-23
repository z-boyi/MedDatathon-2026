# CardioSight — ML-Powered Heart Disease Severity Prediction

> **Machine learning classification pipeline** predicting cardiac disease severity (healthy / mild / moderate-severe) from 920 clinical patient records with 16 features. Applied ensemble methods (LightGBM, XGBoost), handled class imbalance and missing data (66% in coronary arteries, 53% in thalassemia), and used SHAP for feature interpretability.

---

## Team Members

| Name | |
|---|---|
| Xuyang Chen | Xinghao Huang |
| Kaiyue Wang | Yiwen Wang |
| Junyi Yao | Boyi Zhang |

---

## Project Overview

This end-to-end machine learning project investigates whether models can predict heart disease severity using clinical indicators for early intervention and risk stratification. The pipeline covers exploratory data analysis, feature engineering, model training, hyperparameter tuning, and interpretability analysis.

**Key metrics:**
- **Dataset:** 920 patient records, 16 clinical features, 3 target classes
- **Best model:** Tuned LightGBM
- **Primary metric:** Macro F1 score (optimized for multiclass imbalance)
- **Top predictors:** ST depression, chest pain type, age, exercise-induced angina

---

## Project Structure

```
CardioSight/
├── analysis/
│   ├── EDA.ipynb           # Exploratory data analysis with statistical summaries
│   └── modelling.ipynb     # Model training, tuning, evaluation, and SHAP analysis
├── dataset/
│   └── heart_disease.csv   # UCI Heart Disease dataset (920 records, 16 features)
├── visualizations/         # Output artifacts (PNG plots, interactive Altair charts)
├── requirements.txt        # Python dependencies with version pinning
└── README.md              # This file
```

---

## Dataset

**Source:** [UCI Heart Disease Dataset via Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

**Specifications:**
- **Size:** 920 patient records
- **Features:** 16 clinical indicators
- **Target:** `num` mapped to 3 severity classes

| Class | Label | Definition |
|---|---|---|
| 0 | No disease | Healthy baseline |
| 1 | Mild | Mild cardiovascular disease |
| 2 | Moderate–Severe | Significant heart disease requiring intervention |

### Feature Dictionary

| Feature | Type | Range/Values | Description |
|---|---|---|---|
| `age` | Numeric | 29–77 years | Patient age |
| `sex` | Binary | 0, 1 | 0 = female, 1 = male |
| `cp` | Categorical | 1–4 | Chest pain type (4 categories) |
| `trestbps` | Numeric | 94–200 mm Hg | Resting blood pressure |
| `chol` | Numeric | 126–564 mg/dl | Serum cholesterol |
| `fbs` | Binary | 0, 1 | Fasting blood sugar > 120 mg/dl |
| `restecg` | Categorical | 0–2 | Resting electrocardiogram results |
| `thalch` | Numeric | 60–202 bpm | Maximum heart rate achieved |
| `exang` | Binary | 0, 1 | Exercise-induced angina |
| `oldpeak` | Numeric | 0–6.2 | ST depression induced by exercise |
| `slope` | Categorical | 1–3 | Slope of peak exercise ST segment |
| `ca` | Numeric | 0–4 | Major vessels coloured by fluoroscopy (66% missing) |
| `thal` | Categorical | 3–7 | Thalassemia type (53% missing) |

**Data Quality Notes:**
- `ca` (coronary arteries): 611 missing values (~66%)
- `thal` (thalassemia): 486 missing values (~53%)
- Both features retained due to clinical significance; addressed via statistical imputation

---

## Methodology

### Data Preprocessing Pipeline

**Imputation strategies:**
- Numeric features: mean imputation
- Categorical features: most-frequent imputation
- Binary features: most-frequent imputation

**Feature scaling:**
- Numeric features: standardization (zero mean, unit variance)
- Categorical features: one-hot encoding
- Binary features: passthrough encoding

**Train/test split:**
- Stratified 70/30 split
- Train: 644 samples | Test: 276 samples
- Stratification preserves class distribution across splits

### Feature Engineering

Two domain-informed composite features created:

| Feature | Formula | Clinical Rationale |
|---|---|---|
| `risk_score` | `cholesterol + resting_bp` | Combines two primary CVD risk factors |
| `cardiac_stress` | `resting_bp × max_heart_rate` | Captures hemodynamic load during exercise |

### Models Trained & Evaluated

| Model | Type | Purpose |
|---|---|---|
| **DummyClassifier** | Baseline | Stratified random baseline for comparison |
| **Logistic Regression** | Linear model | Linear baseline; surprisingly strong (~0.64 macro F1) |
| **Random Forest** | Ensemble | Captures non-linear patterns, handles feature interactions |
| **XGBoost** | Gradient boosting | Sequential optimization with regularization |
| **LightGBM (tuned)** | Gradient boosting | **Best performer**; optimized hyperparameters via grid search |

**Evaluation methodology:**
- Primary metric: **Macro F1 score** (unweighted average across 3 classes; handles imbalance)
- Cross-validation: stratified k-fold (specific folds in notebooks)
- Hyperparameter tuning: grid search with cross-validation for LightGBM

### Model Results

| Model | Macro F1 Score |
|---|---|
| DummyClassifier (baseline) | ~0.34 |
| Logistic Regression | ~0.64 |
| Random Forest | — |
| XGBoost | — |
| **LightGBM (tuned)** | **Best** |

---

## Key Findings (SHAP Interpretability)

Model interpretation via SHAP (SHapley Additive exPlanations) identified top predictive features:

1. **ST depression (`oldpeak`)** — strongest single predictor
   - Continuous measure of exercise-induced ischemia
   - Clear separation between severity classes

2. **Chest pain type (`cp`)** — symptom classification with high signal
   - Different pain types strongly stratify disease severity
   - Categorical feature with strong discriminative power

3. **Age** — demographic risk factor
   - Older patients skew toward moderate/severe categories
   - Non-linear relationship with severity

4. **Exercise-induced angina (`exang`)** — binary clinical indicator
   - Binary flag indicating chest pain during exertion
   - Strong predictor of disease presence and severity

---

## Visualizations & Artifacts

All outputs saved in `visualizations/`:

- **`eda_plot1–6.png`** — Feature distributions, correlations, severity class separation
- **`eda_plot4–5.html`** — Interactive Altair visualizations for exploration
- **`ml_confusion_matrix.png`** — Confusion matrix for best model (LightGBM)
- **`ml_shap_feature_importance.png`** — SHAP summary plot showing feature contributions

---

## Installation & Usage

### Setup

```bash
# Clone repository
git clone https://github.com/z-boyi/CardioSight.git
cd CardioSight

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Analysis

1. **`analysis/EDA.ipynb`** — Exploratory data analysis
   - Feature distributions, missing data patterns
   - Correlation analysis, severity class visualization
   - Statistical summaries by class

2. **`analysis/modelling.ipynb`** — Model training and evaluation
   - Preprocessing pipeline setup
   - Baseline and production model training
   - Hyperparameter tuning
   - SHAP interpretability analysis
   - Results visualization and comparison

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Data** | pandas, numpy |
| **ML** | scikit-learn, XGBoost, LightGBM |
| **Interpretability** | SHAP |
| **Visualization** | Altair, Matplotlib, Seaborn |
| **Environment** | Jupyter, Python 3.8+ |

---

## Citation

If you use this work, please cite the original dataset:

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease*. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

---

## Key Technologies & Skills Demonstrated

- **Data Science:** EDA, feature engineering, statistical analysis
- **Machine Learning:** Supervised learning, classification, ensemble methods, hyperparameter optimization
- **Data Quality:** Missing data imputation, class imbalance handling, data preprocessing
- **Model Evaluation:** Cross-validation, multiple metrics (Macro F1, accuracy, confusion matrices)
- **Model Interpretability:** SHAP analysis, feature importance
- **Tools & Languages:** Python, pandas, NumPy, scikit-learn, Jupyter, Git
