# Breast Cancer Classification

This project builds and evaluates multiple machine learning models to classify malignant and benign breast cancer cases using the UCI Breast Cancer Wisconsin dataset.  
The notebook performs an extensive hyperparameter tuning and comparison of 11 popular classifiers.

## 📋 Project Overview

The goal of this project is to:
- Load and preprocess the breast cancer dataset
- Train and evaluate multiple machine learning models
- Perform hyperparameter tuning using GridSearchCV
- Select the model with the best performance for breast cancer classification

This serves as a portfolio project to demonstrate supervised machine learning model selection and optimization.

## 📦 Dataset

- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** Built-in from `sklearn.datasets`
- **Features:** 30 numeric features (e.g., radius, texture, perimeter, area, smoothness, etc.)
- **Target:** Binary classification (0 = malignant, 1 = benign)

## 🛠️ Methodology

### Data Preprocessing
- Features scaled using `StandardScaler` to ensure uniform input ranges.

### Models Evaluated
The following algorithms were tested and optimized:
1. Support Vector Classifier (SVC)
2. Decision Tree Classifier
3. Multi-Layer Perceptron (MLPClassifier - Neural Network)
4. Gaussian Naive Bayes
5. Logistic Regression
6. K-Nearest Neighbors
7. Bagging Classifier
8. Random Forest Classifier
9. AdaBoost Classifier
10. Gradient Boosting Classifier
11. XGBoost Classifier

### Hyperparameter Tuning
- Used `GridSearchCV` with 5-fold cross-validation.
- Scored models based on `precision_macro` to balance performance across both classes.

### Model Evaluation
For each model:
- Printed best hyperparameters
- Displayed classification report (precision, recall, f1-score)
- Showed confusion matrix
- Reported precision score on test set

## 💻 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-classification.git
cd breast-cancer-classification

