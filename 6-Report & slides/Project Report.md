# Applying Machine Learning to Heart Disease Prediction
Baseline, Improvement, and Error Analysis

### Dataset link: 
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Team members: 
- Huỳnh Gia An
- Phạm Đức Anh
- Cao Chí Bảo
- Nguyễn Tân Nhật Minh

# 0. Project Overview
### Project Statement
Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible for approximately 17.9 million deaths per year according to the World Health Organization. Early detection of heart disease can significantly improve patient outcomes and reduce mortality. However, clinical diagnosis often relies on expensive tests and specialist consultation, which are not accessible in many healthcare settings.

This project investigates whether a machine learning model — specifically Logistic Regression — can accurately predict the presence of heart disease from routine clinical features such as age, cholesterol level, resting blood pressure, and ECG results. We further investigate whether an LLM (LLaMA 3.1 via Groq) can suggest meaningful feature engineering improvements to the model.

### Motivation
- Heart disease is a preventable condition if detected early.
- This dataset provides a realistic, well-curated benchmark for learning ML classification workflows.

### Real-world Importance
A false negative (missing a true heart disease patient) can be life-threatening. A false positive (incorrectly flagging a healthy patient) leads to unnecessary stress and tests. Therefore, evaluation must go beyond accuracy — we must carefully analyze precision, recall, and the types of errors the model makes.

### ML Task
Binary Classification
- Input: Clinical features of a patient
- Output: HeartDisease = 1 (disease present) or HeartDisease = 0 (disease absent)

### Project Objectives
1. Build a baseline Logistic Regression model on a real clinical dataset.
2. Improve the model through principled hyperparameter tuning (regularization strength C and decision threshold).
3. Conduct error analysis to understand where and why the model fails.
4. Use an LLM to suggest engineered features, then critically evaluate whether those suggestions hold up mathematically and empirically.

### Why this dataset is appropriate
- 918 samples — sufficient for train/test experiments at bachelor level.
- 12 attributes — mix of numerical and categorical, realistic clinical data.
- Balanced classes (~55% positive) — minimal class imbalance issue.
- Well-documented — features are meaningful and interpretable.
- Publicly available — reproducible without ethical barriers.

# 1. Exploratory Data Analysis (EDA)
**DETAIL VERSION & CODE:** [Click here](https://github.com/hgiaan/ML-GroupProject/blob/main/2-EDA%20%26%20Preprocessing/1-EDA.ipynb)

## 1.1. Dataset Overview
- **Initial Shape:** The raw dataset comprises **918 observations** and **12 features**.

### Numerical Features

| Feature | Description | Unit | Notes |
| :--- | :--- | :--- | :--- |
| Age | Patient age | years | Range: 28–77 |
| RestingBP | Resting blood pressure | mm Hg | 0 values likely data errors |
| Cholesterol | Serum cholesterol | mg/dL | 0 values = missing/imputed |
| MaxHR | Maximum heart rate achieved | bpm | Higher = better cardiac function |
| Oldpeak | ST depression (exercise vs rest) | numeric | Higher = more ischemia |

### Categorical Features

| Feature | Description | Values |
| :--- | :--- | :--- |
| Sex | Biological sex | M / F |
| ChestPainType | Type of chest pain | ATA, NAP, ASY, TA |
| FastingBS | Fasting blood sugar > 120 mg/dL | 0 (No) / 1 (Yes) |
| RestingECG | Resting ECG result | Normal, ST, LVH |
| ExerciseAngina | Exercise-induced angina | Y / N |
| ST_Slope | Slope of peak exercise ST segment | Up, Flat, Down |

**Notes:**
- **ChestPainType:** TA (Typical Angina) = classic chest pain; ATA (Atypical Angina) = heart-related but lacking classic symptoms; NAP (Non-Anginal Pain) = not heart-related; ASY (Asymptomatic) = no pain, strongly linked to silent heart disease.
- **RestingECG:** Normal = healthy electrical activity; ST = ST-T wave abnormalities (warning sign); LVH = thickened left heart chamber, often from high blood pressure.
- **ST_Slope:** Up = normal healthy response; Flat = warning sign of ischemia; Down = strong indicator of severe heart disease.

## 1.2. Class Distribution
**Purpose:** Verify class balance before training.

| | |
| :---: | :---: |
| ![image](https://hackmd.io/_uploads/ry-i5d8ezx.png) | ![image](https://hackmd.io/_uploads/HktsqO8eGg.png) |

→ ~55% positive class — slight imbalance, no critical action needed.

## 1.3. Numerical Features (Correlation Heatmap)
**Purpose:** To identify numerical features that are linearly correlated with the target variable (HeartDisease).

![image](https://hackmd.io/_uploads/SyXmsuLlzx.png)

**Insight:**
- **MaxHR (−0.40)** and **Oldpeak (+0.40)** demonstrate the strongest linear relationships with heart disease — establishing them as primary predictive features.
- **Age (+0.28)** and **Cholesterol (−0.23)** display moderate correlations.
- **RestingBP (+0.11)** exhibits the weakest linear correlation.
- No pair of independent features exceeds ±0.70 correlation → no multicollinearity concern for Logistic Regression.

## 1.4. Boxplots: Numerical Features vs Target
**Purpose:** To compare distributions across target classes and visually identify outliers and data anomalies.

![image](https://hackmd.io/_uploads/BJ7knd8ezx.png)

- **Age & Oldpeak:** Patients with heart disease (1) have higher median values.
- **MaxHR:** Patients with heart disease (1) have a noticeably lower median maximum heart rate.
- **Data Anomalies:** Cholesterol shows values down to 0 for class 1; RestingBP has a single data point at 0 — both are biologically impossible.
- **Outliers:** Significant outliers visible in RestingBP, Cholesterol, MaxHR, and Oldpeak → preprocessing required.

## 1.5. Categorical Feature Analysis
**Purpose:** To identify which categorical values are strongest indicators of heart disease.

![image](https://hackmd.io/_uploads/H1LOhOIgfx.png)

- **Sex:** Males exhibit a significantly higher rate of heart disease than females.
- **ChestPainType:** Asymptomatic (ASY) is overwhelmingly associated with heart disease; other types lean toward no disease.
- **FastingBS:** Fasting blood sugar > 120 mg/dL is strongly correlated with heart disease.
- **ExerciseAngina:** Patients with exercise-induced angina (Y) show ~85% heart disease rate.
- **ST_Slope:** Flat and Down slopes are heavily linked to disease; Up slope is predominantly associated with no disease.

## 1.6. Cholesterol Zero-Value Investigation
**Purpose:** To evaluate the nature of zero values in Cholesterol and understand their relationship to the target.

![image](https://hackmd.io/_uploads/SJXRhdUxfe.png)

- **Zeros as Missing Data:** The histogram shows an unnatural spike at exactly 0. Since a human cholesterol level of 0 is biologically impossible, these represent missing values hardcoded as 0 during data collection.
- **Class Imbalance in Missing Data:** 152 missing cholesterol readings occur in heart disease patients vs. only 20 in normal patients — this skew must be handled carefully to avoid introducing bias.

## 1.7. Conclusion
The EDA confirms strong predictors for heart disease, particularly MaxHR, Oldpeak, ExerciseAngina, and ST_Slope. The absence of severe multicollinearity among continuous variables is favorable for Logistic Regression. However, the presence of outliers and impossible zero values makes rigorous preprocessing mandatory.

---

# 2. Data Preprocessing
**DETAIL VERSION & CODE:** [Click here](https://github.com/hgiaan/ML-GroupProject/blob/main/2-EDA%20%26%20Preprocessing/2-Preprocessing.ipynb)

## 2.1. Outlier Removal
A single anomaly where `RestingBP = 0` was identified. This biologically impossible value was removed, reducing the dataset to **917 records** without compromising data integrity.

## 2.2. Categorical Encoding
To convert categorical text data into a machine-readable format:
- **Binary Features (Label Encoding):** `Sex` → M=1, F=0; `ExerciseAngina` → Y=1, N=0; `FastingBS` retained as-is.
- **Multi-Category Features (One-Hot Encoding):** Applied to `ChestPainType`, `RestingECG`, and `ST_Slope` with `drop_first=True`.

The `drop_first=True` parameter drops one dummy column per feature, establishing a reference category and preventing the **Dummy Variable Trap** (perfect multicollinearity) for stable Logistic Regression coefficients.

**Note:** One-hot encoding is applied before splitting since no statistics are learned from the data — only binary columns are created. In production systems, encoding should be fitted on training data only.

## 2.3. Dataset Splitting
The dataset was partitioned into **80% training (733 records)** and **20% testing (184 records)** using a **stratified split** to preserve the proportional class distribution across both subsets.

## 2.4. Missing Value Imputation
Zero values in `Cholesterol` were imputed with the **median** of non-zero training values. Median was chosen over mean because cholesterol distributions are right-skewed, making median more robust to outliers. To prevent **data leakage**, the training median was then applied to the test set.

## 2.5. Feature Scaling
A `StandardScaler` was applied to standardize all features to mean=0, std=1. This is mathematically necessary for Logistic Regression (gradient descent convergence and regularization fairness). The scaler was **fit exclusively on training data** and used to transform both sets.

## 2.6. Conclusion
The resulting `train_processed.csv` and `test_processed.csv` provide a clean, leakage-free foundation for model training.

---

# 3. Model

**DETAIL VERSION & CODE:** [Click here](https://github.com/hgiaan/ML-GroupProject/blob/main/3-Model/Logistic_Regression.ipynb)

## 3.1. Baseline Model (C=1.0, threshold=0.5)

We begin with scikit-learn's default Logistic Regression — regularization strength C=1.0 and decision threshold=0.5.

| Metric | Normal | Heart Disease | Weighted Avg |
| :--- | :---: | :---: | :---: |
| Precision | 0.88 | 0.88 | 0.88 |
| Recall | 0.85 | 0.90 | 0.88 |
| F1-Score | 0.86| 0.89 | 0.88 |
| **Accuracy** | | | **0.8804** |

The baseline already performs well, confirming that Logistic Regression is a strong fit for this dataset. However, in medical prediction, a Recall of 0.90 still means ~10% of actual heart disease patients are missed — which we aim to reduce.

## 3.2. Improved Model

### Choosing Best C via Cross-Validation

We tested C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0} using 5-fold cross-validation, evaluating F1 and Recall on the training set.

| C | CV F1 | CV Recall |
| :---: | :---: | :---: |
| 0.001 | 0.8662 | 0.9210 highest |
| **0.01** | **0.8720 (best)** | 0.8938 |
| 0.1 | 0.8716 | 0.8815 |
| 1.0 | 0.8678 | 0.8765 |
| 10.0 | 0.8664 | 0.8741 |
| 100.0 | 0.8664 | 0.8741 |

![image](https://hackmd.io/_uploads/HkiqsMFlzl.png)

**C=0.01** achieves the highest CV F1-score.   C=0.001 has higher Recall but underfits. → **C=0.01 selected** as the best trade-off.

### Choosing Best Threshold

The default threshold of 0.5 treats both error types equally. In medicine, a **False Negative** (missing a sick patient) is far more dangerous than a **False Positive**. We swept thresholds to maximize Recall while monitoring F1.

| Threshold | Precision | Recall | F1 | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| 0.30 | 0.7656 | 0.9608 Highest | 0.8522 | 0.8152 |
|0.35 | 0.8083 | 0.9510 |  0.8739    | 0.8478
|0.4 | 0.8333 | 0.9314 |0.8796 | 0.8587
| **0.45** | **0.8716** | **0.9314** | **0.9005** | **0.8859** |
| 0.50 | 0.9020 | 0.9020 | 0.9020 | 0.8913 |

Lowering threshold from 0.50 → 0.45: Recall increases +3.3%, F1 drops only −0.0015, Accuracy drops only 0.5%. → **Threshold=0.45 selected.**
![image](https://hackmd.io/_uploads/SkMjhMtlGg.png)

## 3.3. Baseline vs Improved — Final Comparison
![image](https://hackmd.io/_uploads/rkm0nfKgfl.png)
![image](https://hackmd.io/_uploads/Sy6RhfKgGx.png)
![image](https://hackmd.io/_uploads/ryX-TGFxzl.png)


The improved model detects **3% more true heart disease patients**. The small precision drop means slightly more false alarms — an acceptable trade-off in medical prediction where **missing a sick patient is far more costly than a false alarm**.

---

# 4. Evaluation

The model is evaluated on two levels: overall performance and per-class performance for the Heart Disease class.

**Overall:** Both models achieve ~89% accuracy on 184 held-out test cases. The slight accuracy drop in the improved model is intentional — we traded a small amount of precision for a clinically meaningful gain in Recall.

**Heart Disease Class:** The improved model's Recall of 0.9314 means it correctly identifies 93 out of every 100 true heart disease patients. Combined with a F1-Score of 0.9005, the model achieves a strong balance between sensitivity and overall performance.

**Priority of Metrics:**

| Metric | Priority | Reason |
| :--- | :---: | :--- |
| Recall (Heart Disease) | 🔴 Highest | Missing a sick patient (FN) can be life-threatening |
| F1-Score | 🟠 High | Balances precision and recall |
| Accuracy | 🟡 Medium | Overall view, less critical given class imbalance |
| Precision | 🟢 Lower | False alarms cause inconvenience, not harm |

---

# 5. Error Analysis

**DETAIL VERSION & CODE:** [Click here](https://github.com/hgiaan/ML-GroupProject/blob/main/4-ErrorAnalysis/Error_Analysis.ipynb)

**Model analyzed:** Logistic Regression with C=0.01, threshold=0.45

## 5.1. Error Summary Table

| Case | Error Type | Key Features | Root Cause | ML Concept |
| :--- | :--- | :--- | :--- | :--- |
| No. 174 | False Positive | ExerciseAngina=1, Oldpeak=1.0 | Two risk features dominated; protective signals ignored | Overconfidence near decision boundary |
| No. 80 | False Negative | RestingBP=160, Age=59 | High BP underweighted by the model | Feature weight imbalance |
| No. 115 | False Negative | MaxHR=180, ExerciseAngina=0 | High MaxHR masked true disease | Class overlap in feature space |

## 5.2. Case Deep-Dives

### Case 1 — No. 174: False Positive (prob=0.898)
- **Profile:** Age 57, RestingBP 130, Oldpeak 1.0, ExerciseAngina=1 → Predicted: Disease, True: Healthy
- **Clinical view:** ExerciseAngina=1 and Oldpeak=1.0 are individually strong risk signals. However, clinically, these alone do not confirm disease — other protective factors (normal cholesterol, age context) can offset the risk.
- **ML view:** Logistic Regression cannot capture feature interactions. It treats ExerciseAngina and Oldpeak as additive signals without considering conflicting features. The model's high confidence (0.898) reflects a region where training data was dominated by positive cases with similar profiles.

### Case 2 — No. 80: False Negative (prob=0.273)
- **Profile:** Age 59, RestingBP 160, MaxHR 125, Oldpeak 1.0 → Predicted: Healthy, True: Disease
- **Clinical view:** RestingBP=160 mmHg is Stage 2 Hypertension — a well-established cardiovascular risk factor. Combined with age 59, this patient carries significant risk.
- **ML view:** If RestingBP has high variance across both classes (many healthy patients also have elevated BP), its learned coefficient will be relatively low. The model underweights RestingBP compared to features like ST_Slope that more cleanly separate the classes. This reflects a genuine training data limitation, not a model bug.

### Case 3 — No. 115: False Negative (prob=0.311)
- **Profile:** Age 48, MaxHR 168, Oldpeak 1.0, ExerciseAngina=0 → Predicted: Healthy, True: Disease
- **Clinical view:** MaxHR=180 bpm is high for a 48-year-old, which the model interprets as good cardiac fitness. The absence of ExerciseAngina further supports a healthy prediction. This is a real-world case of **silent ischemia** — disease present without typical exercise symptoms.
- **ML view:** This patient sits in a region of overlapping classes — features that are statistically more common in healthy patients. No linear model can perfectly separate these cases. Lowering the threshold further could catch this case but would introduce more false positives.

## 5.3. Error Rate by Sex

![image](https://hackmd.io/_uploads/ryjgAztxzg.png)

The test set contains significantly more male cases than female cases. This means the model has seen fewer female examples during training, which may affect its reliability for that group. Additionally, female heart disease is known clinically to present with more atypical symptoms — which can go either way in a data-driven model.

## 5.4. What These Errors Reveal About the Model

1. **Logistic Regression cannot capture feature interactions.** When features push predictions in opposite directions, the model simply adds their contributions — it cannot reason about which signal is more reliable in context.
2. **The decision boundary is fixed and linear.** Cases near this boundary are inherently uncertain, and the three cases above all fall in this difficult region.
3. **Missing clinical features limit the model.** Family history, BMI, smoking status, and medication use are absent from this dataset. Some misclassifications may be unavoidable given what the model was given.

---

# 6. LLM Component

**DETAIL VERSION & CODE:** [Click here](https://github.com/hgiaan/ML-GroupProject/tree/main/5-API)

## 6.1. Approach

We queried **LLaMA 3.1 (8B)** via the Groq API with a structured prompt containing our exact 15 feature names, their types (all Z-score standardized), and the target variable. We explicitly stated the preprocessing constraint and asked for 3 clinically relevant engineered features.

## 6.2. LLM Suggestions & Evaluation

| # | Suggested Feature | Clinical Rationale | Decision | Reason |
| :---: | :--- | :--- | :---: | :--- |
| 1 | `Age × Risk Score` | Age combined with multiple risk indicators represents compounded risk | ❌ Rejected | Z-score × Z-score: when Age is negative and Risk is positive, the product flips sign — a young high-risk patient appears lower-risk than a young low-risk patient |
| 2 | `log(RestingBP / Cholesterol)` | High BP:Cholesterol ratio indicates cardiovascular strain | ❌ Rejected | Both are Z-score scaled → denominator can be zero or negative → log of a negative number is undefined |
| 3 | `Oldpeak + (ExerciseAngina × ST_Slope_Up)` | Combines ST depression with exercise-induced angina and slope — a composite stress test index | ✅ Accepted | Addition is safe on Z-scores; multiplying two binary Z-scores is less harmful; clinically meaningful |

## 6.3. Implementation

The accepted feature was added to both train and test sets before retraining:

```
Exercise_Stress_Index = Oldpeak + (ExerciseAngina × ST_Slope_Up)
```

## 6.4. Before vs After Comparison
![image](https://hackmd.io/_uploads/SJrvCzKlGe.png)


| Metric | Before | After | Δ |
| :--- | :---: | :---: | :---: |
| Accuracy | 0.8859 | 0.8696 | −0.0163 |
| Precision | 0.8716 | 0.8482 | −0.0234 |
| Recall | 0.9314 | 0.9314 | 0.0000 |
| F1-Score| 0.9005 | 0.8879 | −0.0126 |

The confusion matrix confirms it as well: Before → 68 TN, 14 FP, 7 FN, 95 TP. After → 65 TN, 17 FP, 7 FN, 95 TP. The number of false negatives (FN) remained unchanged, but false positives (FP) increased by 3. This indicates that the feature did not provide any benefit and actually caused the model to misclassify more healthy patients.

**The feature did not improve performance — it actively degraded the model.**

## 6.5. Why the Feature Failed

**1. Mathematical distortion of standardized space.**
The formula multiplies two Z-score standardized binary values (ExerciseAngina and ST_Slope_Up). Because Z-scores include negative values for patients below the mean, multiplying two negatives produces a positive product — destroying the directional meaning that Logistic Regression relies on.

**2. Feature redundancy.**
The index aggregates Oldpeak, ExerciseAngina, and ST_Slope_Up — all of which already exist independently in the dataset. Logistic Regression inherently learns to weight these features optimally on its own. Forcing them into a single interaction term adds noise rather than novel information.

## 6.6. Conclusion: What This Tells Us About LLM-Assisted Feature Engineering

- **The LLM served as a strong clinical advisor.** It correctly identified that stress test components are medically interconnected and formulated a sound clinical hypothesis.
- **However, the LLM lacked mathematical and pipeline context.** It failed to account for the pre-applied Z-score standardization and the additive mechanics of Logistic Regression.

> **LLMs are highly effective for generating domain-specific hypotheses, but their suggestions must be strictly evaluated against the mathematical realities of the specific data pipeline. Clinical validity does not guarantee computational effectiveness.**

# 7. Conclusion  
This project demonstrated a full supervised ML pipeline — from EDA to preprocessing, modeling, error analysis, and LLM-assisted feature engineering — applied to a real clinical dataset.

The improved Logistic Regression (C=0.01, threshold=0.45) achieved **88.6% accuracy** and **93.1% Recall** on the Heart Disease class, meaningfully reducing the rate of missed diagnoses compared to the baseline.
The LLM experiment produced an important negative result: the suggested feature was clinically sound but mathematically incompatible with a Z-score standardized pipeline. This highlights a key lesson — domain knowledge from an LLM is a starting point, not a final answer. Every suggestion must be validated against the actual data pipeline before implementation.

**Limitations:**

- Logistic Regression assumes linearity - it cannot capture complex feature interactions.
- Several clinically relevant features (BMI, smoking status, family history) are absent from the dataset, which likely explains some irreducible errors.
- The small dataset size (917 records) limits generalizability.

**Future Work:**

- Test non-linear models (Random Forest, XGBoost) to handle feature interactions.
- Collect richer clinical features to reduce the false negative rate.
- Explore better prompting strategies to make the LLM aware of preprocessing constraints before suggesting features.

**Thank you for reading!**
