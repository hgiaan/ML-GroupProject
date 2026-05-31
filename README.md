# Group Project: _Can AI Help You Build a Better Machine Learning Model?_
# Heart Disease Prediction — ML Group Project

> **Can AI Help You Build a Better Machine Learning Model?**  
> Binary classification using Logistic Regression to predict heart disease, with LLM-assisted feature engineering.

---

## 📌 Project Overview

This project explores whether AI (specifically a Large Language Model) can meaningfully improve a traditional Machine Learning pipeline. We build a Logistic Regression model to predict heart disease from clinical data, then use **LLaMA 3.1 via Groq** to suggest novel engineered features — and critically evaluate whether those suggestions are statistically and clinically sound.

**Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (UCI, via Kaggle)  
**Task:** Binary classification — predict `HeartDisease` (0 = Normal, 1 = Heart Disease)

---

## 🔍 Features

- **Exploratory Data Analysis (EDA):** Distribution plots, correlation heatmaps, class balance checks, and outlier detection.
- **Data Preprocessing:** Handling missing values, encoding categorical variables, Z-score standardization (StandardScaler).
- **Logistic Regression Model:** Trained and evaluated with standard classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
- **Error Analysis:** Investigating false positives and false negatives to understand model failure modes.
- **LLM-Assisted Feature Engineering:** Querying LLaMA 3.1 for clinically relevant feature suggestions, then rigorously evaluating each one.
- **Before vs. After Comparison:** Quantifying whether LLM-suggested features actually improve model performance.

---

## 🛠️ Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML / Stats | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn |
| LLM Integration | LLaMA 3.1 via [Groq API](https://groq.com/) |
| Notebooks | Jupyter Notebook |

---

## 📂 Project Structure

```
ML-GroupProject/
├── 0-Brainstorms & Implementation/   # Planning docs and design decisions
├── 1-Data/
│   └── heart.csv                     # Raw dataset (download from Kaggle)
├── 2-EDA & Preprocessing/
│   ├── 1-EDA.ipynb                   # Exploratory Data Analysis
│   └── 2=Preprocessing.ipynb         # Cleaning, encoding, scaling → outputs CSVs
├── 3-Model/
│   └── Logistic_Regression.ipynb    # Model training and evaluation
├── 4-ErrorAnalysis/
│   └── Error_Analysis.ipynb         # Analysis of model errors
├── 5-API/                           # (Optional) API deployment scripts
│   └── 1-llm_feature_engineering.ipynb  # LLM query + feature evaluation
│   └── 2-Before_and_After_models_comparison.ipynb  # Performance comparison
├──6-Report & slides
│   └── Project_Report.md
│   └── Heart_Disease_Prediction_ML_Project.pdf
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hgiaan/ML-GroupProject.git
cd ML-GroupProject
```

### 2. Install dependencies

```bash
python --version # must be >=3.10
pip install -r requirements.txt
```

### 3. Download the dataset

Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and place it in the `1-Data/` folder.

### 4. Set up Groq API key (for LLM notebook)

```bash
export GROQ_API_KEY="your_key_here"
```

---

## ▶️ Run Order (Pipeline)

Run the notebooks **in this exact order**:

| Step | Notebook | Description |
|---|---|---|
| 1 | `2-EDA & Preprocessing/1-EDA.ipynb` | Explore the raw dataset |
| 2 | `2-EDA & Preprocessing/2-Preprocessing.ipynb` | Generate cleaned/scaled CSVs |
| 3 | `3-Model/Logistic_Regression.ipynb` | Train and evaluate baseline model |
| 4 | `4-ErrorAnalysis/Error_Analysis.ipynb` | Analyze errors on baseline model |
| 5 | `5-API/1-llm_feature_engineering.ipynb` | Query LLM and evaluate suggested features |
| 6 | `5-API/2-Before_and_After_models_comparison.ipynb` | Compare baseline vs. LLM-enhanced model |

---

## 🤖 AI Usage

We used **LLaMA 3.1 (8B Instant)** via the Groq API to suggest clinically relevant engineered features for our Logistic Regression model.

The LLM proposed 3 features:

| Feature | Clinical Rationale | Team Decision |
|---|---|---|
| `Age × Risk Score` | Age + compounded risk indicators | ❌ **Rejected** — Z-score × Z-score produces unstable, misleading values |
| `log(RestingBP / Cholesterol)` | High BP:Cholesterol ratio = cardiovascular strain | ❌ **Rejected** — Division of Z-scores can produce undefined/negative log values |
| `Oldpeak + (ExerciseAngina × ST_Slope_Up)` | Combines stress test results into a single diagnostic index | ✅ **Accepted** — Mathematically safe; addition dominates; clinically meaningful |

This critical evaluation process is central to the project's research question: **AI suggestions must be validated, not blindly applied.**

---

## 👥 Team Members

> Huỳnh Gia An
> Phạm Đức Anh
> Cao Chí Bảo
> Nguyễn Tân Nhật Minh

---

## 📄 License

This project is for academic purposes. Dataset is sourced from the UCI Machine Learning Repository via Kaggle.