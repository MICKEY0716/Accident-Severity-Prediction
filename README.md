# 🚦 Accident Severity Prediction

> **Machine Learning-Based Classification for Road Safety Risk Assessment**

---

## 📌 Overview

Road traffic accidents pose a significant risk to public safety and infrastructure.  
This project focuses on predicting **accident severity** using machine learning techniques based on historical accident-related data.

The objective is to identify high-risk scenarios and support **data-driven safety interventions** by accurately classifying accident severity levels.

---

## 🎯 Problem Statement

Accident severity is influenced by multiple factors such as:
- Time of accident
- Road and environmental conditions
- Victim characteristics
- Safety equipment usage

Without predictive systems, it becomes difficult to proactively identify and mitigate high-risk situations.

**Objective:**  
Build a robust classification model that predicts accident severity and helps identify critical risk factors.

---

## 📊 Dataset Overview

- Structured accident-related dataset
- Includes temporal, behavioral, and safety-related attributes
- Highly imbalanced target variable

> Dataset details are kept concise to emphasize modeling strategy and evaluation.

---

## 🧠 Approach & Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Severity distribution analysis
- Correlation and feature importance exploration
- Identification of imbalance in target classes

### 2️⃣ Feature Engineering
- Time-based feature extraction
- Risk-related feature creation
- Handling categorical and binary indicators

### 3️⃣ Handling Class Imbalance
- Focused evaluation beyond accuracy
- Recall-oriented optimization for severe cases
- Decision threshold tuning

### 4️⃣ Model Development
- Trained multiple classification models
- Selected final model based on balanced performance
- Optimized decision threshold for real-world applicability

---

## 🤖 Model Used

### 🔹 Random Forest Classifier

Chosen for:
- Robustness on structured data
- Ability to capture non-linear relationships
- Interpretability via feature importance

---

## 📈 Evaluation & Results

- ROC-AUC used as primary evaluation metric
- High recall achieved for severe accident cases
- Threshold tuning improved real-world sensitivity
- Model prioritizes safety-critical predictions

📌 Detailed metrics and analysis are available in the notebook.

---

## 🖥️ User Interface

An interactive user interface is included to:
- Accept accident-related inputs
- Predict severity level
- Improve accessibility for non-technical users

This enables practical usage beyond model experimentation.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Scikit-learn  
- **Deployment:** Streamlit / Flask  
- **Model Persistence:** Pickle  

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```
© 2025 Rachit Patwa. All rights reserved.
