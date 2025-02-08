
# Predictive Analytics for Diabetes Risk Assessment

## Overview

This project implements multiple machine learning models to predict diabetes risk based on the Behavioral Risk Factor Surveillance System (BRFSS) health indicators dataset. The models used include Decision Tree, Gaussian Naive Bayes, Logistic Regression, and Random Forest classifiers. The dataset undergoes preprocessing, feature scaling, data balancing using SMOTE, and hyperparameter tuning for improved performance.

## Dataset

The dataset is loaded from transformed_diabetes_data_v2.csv.

It contains health indicators such as High Blood Pressure, Cholesterol Check, Smoking Status, Heart Disease History, Physical Activity, General Health Status, Age, and BMI category.

The target variable is Diabetes_012, which classifies individuals as:

0: No Diabetes

1: Prediabetes

2: Diabetes

### Data Preprocessing

- Features and target variable are separated.

- Standardization is applied using StandardScaler.

- Data is split into training (70%), validation (15%), and test (15%) sets.

- The SMOTE technique is applied to balance the training dataset.

## Machine Learning Models

1. **Decision Tree Classifier**

Hyperparameter tuning using GridSearchCV.

Evaluated on training, validation, and test sets.

AUC-ROC score and confusion matrices are generated.

2. **Gaussian Naive Bayes**

Uses a preprocessing pipeline with StandardScaler.

Trained on the resampled dataset.

Evaluated using classification reports, confusion matrices, and ROC curves.

Cross-validation accuracy is calculated.

3. **Logistic Regression**

Features are normalized using MinMaxScaler.

Hyperparameter tuning performed with RandomizedSearchCV.

Model evaluation includes accuracy scores, confusion matrices, and AUC-ROC analysis.

4. **Random Forest Classifier**

Hyperparameter tuning using GridSearchCV with StratifiedKFold.

Evaluated with classification reports, confusion matrices, and AUC-ROC.

Feature importance analysis is performed to understand key predictors.

## Performance Metrics

For each model:

- Accuracy

- Confusion Matrix

- Classification Report

- AUC-ROC Curve

- Cross-validation accuracy (where applicable)

- Visualization

- ROC Curves for model comparison.

- Feature importance analysis for Random Forest.

## Prerequisites

Required Libraries:

- Install dependencies using the following command:

- pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost scipy

### Usage

- Ensure the dataset is placed in the specified directory.

- Run the script to preprocess data and train models.

- Review model performance metrics in the console output.

- Use ROC curves and feature importance analysis to interpret results.

## Authors

- [@Andres De Los Santos]()
- [@Gilber Hernandez](https://www.github.com/gilbermhb)

## Acknowledgments

This project is part of the Data Science Master's Capstone at Fordham University.
