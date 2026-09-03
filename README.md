# Employee Turnover Prediction Using Logistic Regression

## 📌 Project Overview

Employee turnover is an important challenge for organizations because losing employees can increase recruitment costs, reduce productivity, and affect team performance.

This project uses **Machine Learning classification techniques** to predict whether an employee is likely to leave the organization.

The project implements **Logistic Regression** with three different approaches:

* **Baseline Logistic Regression**
* **L1 Regularization (Lasso)**
* **L2 Regularization (Ridge)**

The models are trained and evaluated using an employee dataset containing **1,350 records and 15 input features**.

---

## 🎯 Objectives

The main objectives of this project are:

* Predict employee turnover using machine learning.
* Explore Logistic Regression for binary classification.
* Compare regularized and non-regularized Logistic Regression models.
* Understand the impact of **L1 (Lasso)** and **L2 (Ridge)** regularization.
* Evaluate model performance using accuracy, precision, recall, and F1-score.

---

## 📂 Project Structure

```text
Employee-Turnover-Prediction/
│
├── employee_turnover.csv
├── employee_turnover_project.ipynb
└── README.md
```

---

## 📊 Dataset

The dataset contains **1,350 employee records** with 15 predictor variables and one target variable.

### Features

| Feature                                   | Description                                         |
| ----------------------------------------- | --------------------------------------------------- |
| `Job_Satisfaction`                        | Employee job satisfaction level                     |
| `Performance_Rating`                      | Employee performance rating                         |
| `Years_At_Company`                        | Number of years at the company                      |
| `Work_Life_Balance`                       | Work-life balance measure                           |
| `Distance_From_Home`                      | Distance of employee's home from workplace          |
| `Monthly_Income`                          | Monthly employee income                             |
| `Education_Level`                         | Employee education level                            |
| `Age`                                     | Employee age                                        |
| `Num_Companies_Worked`                    | Number of companies previously worked for           |
| `Employee_Role`                           | Encoded employee role                               |
| `Annual_Bonus`                            | Annual bonus value                                  |
| `Training_Hours`                          | Training hours                                      |
| `Department`                              | Encoded department                                  |
| `Annual_Bonus_Squared`                    | Squared annual bonus feature                        |
| `Annual_Bonus_Training_Hours_Interaction` | Interaction between annual bonus and training hours |

### Target Variable

`Employee_Turnover`

* `0` → Employee does not leave
* `1` → Employee leaves

> **Note:** The categorical-looking variables in this dataset are already represented as numerical values, so the notebook directly uses them as model inputs.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data loading and manipulation
* **Scikit-learn** – Machine learning and evaluation
* **Jupyter Notebook** – Development environment
* **Logistic Regression** – Classification algorithm

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Load Data using Pandas
   ↓
Separate Features (X) and Target (y)
   ↓
Train-Test Split
   ↓
Train Logistic Regression Models
   ├── Baseline Logistic Regression
   ├── L1 Regularization (Lasso)
   └── L2 Regularization (Ridge)
   ↓
Make Predictions
   ↓
Evaluate Models
   ↓
Compare Performance
```

---

## 🧠 Models Implemented

### 1. Baseline Logistic Regression

A standard Logistic Regression model is used as the baseline.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
```

---

### 2. L1 Regularization — Lasso

L1 regularization can reduce the influence of less important features by shrinking coefficients toward zero.

```python
lasso = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=0.5
)

lasso.fit(X_train, y_train)
```

---

### 3. L2 Regularization — Ridge

L2 regularization penalizes large model coefficients and can help control model complexity.

```python
ridge = LogisticRegression(
    penalty='l2',
    C=1,
    max_iter=200
)

ridge.fit(X_train, y_train)
```

---

## 🧪 Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This results in:

* **Training data:** 1,080 records
* **Testing data:** 270 records

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## 🏆 Results

The models produced the following test-set accuracy:

| Model                        |   Accuracy |
| ---------------------------- | ---------: |
| Baseline Logistic Regression | **85.93%** |
| L1 Regularization (Lasso)    | **87.04%** |
| L2 Regularization (Ridge)    | **85.93%** |

### Best Model

The **L1-Regularized Logistic Regression (Lasso)** achieved the highest accuracy:

> **87.04%**

For the Lasso model:

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| 0     |      0.86 |   0.90 |     0.88 |
| 1     |      0.88 |   0.83 |     0.86 |

Overall:

* **Accuracy:** 87.04%
* **Macro F1-score:** 0.87
* **Weighted F1-score:** 0.87

---

## 🔍 Key Findings

1. Logistic Regression provides a strong baseline for predicting employee turnover.
2. The **L1-regularized model performed best** among the three approaches tested.
3. L1 regularization improved test accuracy from approximately **85.93% to 87.04%**.
4. The Lasso model achieved a recall of **0.83 for employees who leave**, meaning it correctly identified a substantial portion of turnover cases.
5. The baseline and Ridge models produced the same test accuracy in this experiment.

---

## 💡 Business Use Case

An employee turnover prediction model can help organizations identify employees who may be at higher risk of leaving.

Organizations could potentially use such predictions to:

* Identify retention risks.
* Improve employee engagement.
* Review job satisfaction.
* Analyze work-life balance.
* Improve training programs.
* Develop employee retention strategies.
* Reduce unexpected employee turnover.

> This project demonstrates the machine-learning approach; predictions should not be treated as the sole basis for employment decisions.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Employee-Turnover-Prediction
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the Notebook

Open:

```text
employee_turnover_project.ipynb
```

Make sure `employee_turnover.csv` is located in the same directory as the notebook.

---

## 📦 Requirements

```text
Python 3.x
pandas
scikit-learn
jupyter
```

---

## 📚 Concepts Demonstrated

This project demonstrates practical understanding of:

* Supervised Machine Learning
* Binary Classification
* Logistic Regression
* Train-Test Split
* L1 Regularization
* L2 Regularization
* Model Training
* Model Prediction
* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Model Comparison
* Employee Turnover Prediction

---

## 🔮 Future Improvements

The project can be further improved by:

* Performing Exploratory Data Analysis (EDA).
* Checking feature distributions and correlations.
* Applying feature scaling where appropriate.
* Using cross-validation.
* Performing hyperparameter tuning.
* Comparing additional classification algorithms such as Random Forest, XGBoost, SVM, and Decision Trees.
* Evaluating ROC-AUC and PR-AUC.
* Creating a confusion matrix.
* Performing feature importance/coefficients analysis.
* Building an interactive prediction interface using Streamlit.
* Deploying the trained model as a web application or API.

---

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech – Computer Science Engineering (Artificial Intelligence & Machine Learning)

---

## ⭐ Project Summary

This project demonstrates how **Logistic Regression and regularization techniques** can be applied to an employee dataset to predict employee turnover.

Among the tested approaches, **L1-Regularized Logistic Regression achieved the best performance with 87.04% test accuracy**.

The project provides a practical foundation for understanding classification, regularization, model evaluation, and applying machine learning to a real-world business problem.
