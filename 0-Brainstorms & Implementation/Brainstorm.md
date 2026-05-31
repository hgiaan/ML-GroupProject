# 0 — Brainstorming & Implementation

## Research Question
**Can AI help you build a better machine learning model?**

---

## Idea

- Task: Heart disease prediction (binary classification)
- Model: Logistic Regression — simple, easy to control, and makes it easier to observe the impact of newly engineered features.
- Dataset: UCI Heart Disease (Kaggle)
- LLM: LLaMA 3.1 via Groq API

The reason for choosing Logistic Regression instead of a more complex model is to isolate the effect of feature engineering and avoid confounding it with model complexity.
Use an LLM to suggest feature engineering ideas for a machine learning model, then test whether those features actually improve performance.

---

## Plan

1. Perform EDA to understand the data, identify outliers, and examine class balance.
2. Preprocess the data (cleaning, encoding, scaling, and train-test splitting).
3. Train a baseline Logistic Regression model.
4. Conduct error analysis to identify where the model makes mistakes.
5. Ask the LLM for feature engineering suggestions and evaluate each proposed feature.
6. Compare model performance before and after adding the LLM-generated features.

---

## LLM Feature Ideas (Raw)

The LLM suggested three candidate features:

- `Age × Risk Score` → Rejected (multiplying two Z-scores produces a feature with unclear interpretation).
- `log(RestingBP / Cholesterol)` → Rejected (dividing standardized values can produce negative ratios, making the logarithm undefined).
- `Oldpeak + (ExerciseAngina × ST_Slope_Up)` → **Selected** (the operation is mathematically safe and has a plausible clinical interpretation).

---

## Team role
| Member | Responsibility |
|---|---|
| Huỳnh Gia An | EDA & Preprocessing |
| Phạm Đức Anh | Model Training |
| Cao Chí Bảo | Error Analysis |
| Nguyễn Tân Nhật Minh | LLM Integration & Evaluation |