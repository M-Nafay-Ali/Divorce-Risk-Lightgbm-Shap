# 💍 Marital Dynamics & Relationship Risk Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://divorce-risk-lightgbm-shap.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-red)

> **Portfolio Project #21** | Engineered, evaluated, and deployed entirely on a **mobile device** using Google Colab and web interfaces.

An end-to-end production machine learning application and interpretability engine built to assess marital stability and divorce risk using regularized gradient boosting (LightGBM) and TreeSHAP explainability mechanics.

🌐 **Live Interactive App:** [divorce-risk-lightgbm-shap.streamlit.app](https://divorce-risk-lightgbm-shap.streamlit.app/)

---

## 📌 Executive Summary & Motivation

Understanding marital stability requires looking beyond simple demographic markers to analyze complex, non-linear psychological interactions. This project uses Gottman Institute behavioral indicators (e.g., contempt, criticism, stonewalling, defensiveness) alongside socioeconomic factors to predict divorce outcomes while maintaining **100% decision transparency** via exact SHAP values.

---

## 🛠️ Machine Learning Pipeline & Architecture

1. **Data Leakage Elimination:** Dropped target-dependent variables (`years_to_divorce`, `years_married`) and unique metadata IDs before pipeline fitting.
2. **Feature Preprocessing via `ColumnTransformer`:**
   * **`OrdinalEncoder`:** Encoded `education_level` preserving natural rank (`less_than_hs` < `high_school` < `some_college` < `bachelors` < `graduate`).
   * **`OneHotEncoder`:** Encoded nominal features (`religious_attendance`), dropping first dummy variables to prevent collinearity.
3. **Model Selection & Regularization:** Trained a `LGBMClassifier` (`n_estimators=100`, `learning_rate=0.05`, `max_depth=5`) embedded directly in a scikit-learn `Pipeline`.

---

## 📊 Model Performance & Error Analysis

Evaluated on a $20\%$ held-out test split ($9,000$ samples):

* **Train ROC-AUC:** `0.8548`
* **Test ROC-AUC:** `0.8333` *(Minimal ~0.02 gap confirms robust generalization without overfitting)*
* **Test Accuracy:** ~`75.4%`
* **Macro Average F1-Score:** `0.75`

<p align="center">
  <img src="assets/confusion_matrix.png" alt="Test Set Confusion Matrix" width="450"/>
</p>

---

## 🔍 Explainable AI & SHAP Interpretability

### 1. Global Feature Importance (SHAP Summary Plot)
<p align="center">
  <img src="assets/shap_summary.png" alt="SHAP Summary Plot" width="600"/>
</p>

* **Primary Driver of Risk:** **Contempt** emerged as the single strongest predictor of marital failure, with high values (magenta cluster) driving log-odds contributions upwards of $+1.2$.
* **Protective Buffers:** High `repair_attempt_success` and elevated `positive_negative_ratio` consistently pull predictions toward the negative side (reducing risk).

---

### 2. Local Instance Explanation (SHAP Waterfall Plot)
<p align="center">
  <img src="assets/shap_waterfall.png" alt="SHAP Waterfall Plot" width="650"/>
</p>

Deconstructs an individual prediction ($f(x) = -1.121$) against baseline log-odds ($E[f(X)] = -0.209$):
* **Risk Drivers (+):** High `criticism` ($+0.39$) and elevated `contempt` ($+0.32$) pull the score toward high risk.
* **Protective Drivers (-):** Older `age_at_marriage` ($-0.35$), higher `education_level` ($-0.29$), and first marriage status ($-0.24$) override risk factors to correctly classify the couple as stable (`0`).

---

### 3. Non-Linear Feature Interaction (SHAP Dependence Plot)
<p align="center">
  <img src="assets/shap_dependence.png" alt="SHAP Dependence Plot" width="550"/>
</p>

* Demonstrates a monotonic protective effect as `education_level` increases from $0$ to $4$.
* Highlights a clear risk crossover threshold at level $2$ (`some_college`), above which education consistently reduces divorce risk.

---

## 💻 Tech Stack & Dependencies

* **Language & Runtime:** Python
* **Machine Learning:** `scikit-learn==1.6.1`, `lightgbm`, `joblib`
* **Explainability:** `shap`, `matplotlib`
* **Dashboard & Visuals:** `streamlit`, `plotly`

---

## 📱 Mobile-First Development Note

This project was developed end-to-end—from data ingestion, EDA, pipeline construction, hyperparameter tuning, model serialization, to GitHub repository setup and Streamlit deployment—entirely on a **mobile device**.


## 📞 Contact Information:-
* **Email:-**[englandengland271@gmail.com]
* **Linkedin:-**[https://www.linkedin.com/in/mohammed-nafay-ali-16519138a?utm_source=share_via&utm_content=profile&utm_medium=member_android]
* **GitHub:-**[https://github.com/M-Nafay-Ali]
