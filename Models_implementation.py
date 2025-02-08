# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:48:14 2024

@author: Gilbert Hernandez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler ,label_binarize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from scipy.stats import uniform, randint



print(
      "Model Implementation - Predictive Analytics for Diabetes Risk Assessment Using Machine Learning on BRFSS Health Indicator"
      )

data = pd.read_csv('C:/Users/Gilbert Hernandez/OneDrive - Fordham University/Capstone/transformed_diabetes_data_v2.csv')

df = data[['HighBP','HighChol','CholCheck','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','HvyAlcoholConsump','NoDocbcCost','GenHlth','DiffWalk','Sex','Age_Mapped','BMI_Status_Number','Diabetes_012']]


#Defining X and y (target variable)
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# First split: training + validation (85%) and test (15%)
X_main, X_test, y_main, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)

# Second split: training (85% of training + validation) and validation (15% of training + validation)
X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.15, random_state=42, stratify=y_main)

""" 
Final:
Train --> 70%
Validation --> 15%
Test --> 15%
"""


# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)






print(
      """ ### Decision Tree ## """
)


# Initialize the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'max_depth': [5],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'criterion': ['entropy']
}


# Initialize GridSearchCV with the Decision Tree classifier and parameter grid
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best estimator
best_clf = grid_search.best_estimator_

# Print the best parameters
print('Best parameters found by GridSearchCV:')
print(grid_search.best_params_)

# Train the best estimator on the balanced training data
best_clf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model on the training set
y_train_pred = best_clf.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Print training classification report
print('Training Classification Report:')
print(classification_report(y_train_resampled, y_train_pred))

# Print training confusion matrix
print('Training Confusion Matrix:')
print(confusion_matrix(y_train_resampled, y_train_pred))

# Evaluate the model on the validation set
y_val_pred = best_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Print validation classification report
print('Validation Classification Report:')
print(classification_report(y_val, y_val_pred))

# Print validation confusion matrix
print('Validation Confusion Matrix:')
print(confusion_matrix(y_val, y_val_pred))

# Evaluate the model on the test set
y_test_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Print test classification report
print('Test Classification Report:')
print(classification_report(y_test, y_test_pred))

# Print test confusion matrix
print('Test Confusion Matrix:')
print(confusion_matrix(y_test, y_test_pred))

# Calculate AUC-ROC for the test set
y_test_prob = best_clf.predict_proba(X_test)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f'AUC-ROC Score: {roc_auc:.2f}')






print(
      """ ### Gaussian Naive Bayes ## """
)

#Specifying features of interest
selected_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
                     'HeartDiseaseorAttack', 'PhysActivity', 'HvyAlcoholConsump',
                     'NoDocbcCost', 'GenHlth', 'DiffWalk', 'Sex', 'Age_Mapped',
                     'BMI_Status_Number']

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), selected_features)
    ]
)

# Initialize the pipeline with predefined Naive Bayes parameters
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('nb', GaussianNB(var_smoothing=1e-9))  # Manually set var_smoothing
])


# Train the model on the balanced training data
pipeline.fit(X_train_resampled, y_train_resampled)

# Evaluate the model on the training set
y_train_pred = pipeline.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Print training classification report
print('Training Classification Report:')
print(classification_report(y_train_resampled, y_train_pred))

# Print training confusion matrix
print('Training Confusion Matrix:')
print(confusion_matrix(y_train_resampled, y_train_pred))

# Evaluate the model on the validation set
y_val_pred = pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Print validation classification report
print('Validation Classification Report:')
print(classification_report(y_val, y_val_pred))

# Print validation confusion matrix
print('Validation Confusion Matrix:')
print(confusion_matrix(y_val, y_val_pred))

# Evaluate the model on the test set
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Print test classification report
print('Test Classification Report:')
print(classification_report(y_test, y_test_pred))

# Print test confusion matrix
print('Test Confusion Matrix:')
print(confusion_matrix(y_test, y_test_pred))

# Calculate AUC-ROC for the test set
y_test_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f'AUC-ROC Score: {roc_auc:.2f}')

# Evaluate the model using cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')









print(
      """ ### Logistic Regression ## """
)

# Initialize the MinMaxScaler and normalize the features
scaler = MinMaxScaler()
X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression classifier
clf = LogisticRegression(random_state=42, max_iter=1000)

# Define the parameter grid for RandomizedSearchCV
param_distributions = {
    'penalty': ['l1'],
    'C': uniform(0.08066305219717405),
    'solver': ['liblinear'],
    'class_weight': [None]
}

# Initialize RandomizedSearchCV with the Logistic Regression classifier and parameter grid
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_distributions, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train_resampled_scaled, y_train_resampled)

# Get the best estimator
best_clf = random_search.best_estimator_

# Print the best parameters
print(f'Best Parameters: {random_search.best_params_}')

# Train the best estimator on the full resampled training data
best_clf.fit(X_train_resampled_scaled, y_train_resampled)

# Evaluate the model on the training set
y_train_pred = best_clf.predict(X_train_resampled_scaled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Print training classification report
print('Training Classification Report:')
print(classification_report(y_train_resampled, y_train_pred))

# Print training confusion matrix
print('Training Confusion Matrix:')
print(confusion_matrix(y_train_resampled, y_train_pred))

# Evaluate the model on the validation set
y_val_pred = best_clf.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Print validation classification report
print('Validation Classification Report:')
print(classification_report(y_val, y_val_pred))

# Print validation confusion matrix
print('Validation Confusion Matrix:')
print(confusion_matrix(y_val, y_val_pred))

# Evaluate the model on the test set
y_test_pred = best_clf.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Print test classification report
print('Test Classification Report:')
print(classification_report(y_test, y_test_pred))

# Print test confusion matrix
print('Test Confusion Matrix:')
print(confusion_matrix(y_test, y_test_pred))

# Calculate AUC-ROC for the test set
y_test_prob = best_clf.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

print(f'AUC-ROC Score: {roc_auc:.2f}')

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_clf, X_train_resampled_scaled, y_train_resampled, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}')









print(
      """ ### Random Forest ## """
)

# # Define class weights
class_weights = {0: 1, 1: 2}

# Random Forest Classifier
rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

# GridSearchCV with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='accuracy')

# Training the data
grid_search.fit(X_train_resampled, y_train_resampled)

# Print the best parameters found by grid search
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(X_train_resampled, y_train_resampled)

# Evaluate the model on the training set
y_train_pred = best_rf_model.predict(X_train)
y_train_pred_proba = best_rf_model.predict_proba(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
train_class_report = classification_report(y_train, y_train_pred)

# Compute ROC AUC scores for each class
y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
train_roc_auc = {}
for i in range(len(np.unique(y))):
    train_roc_auc[i] = roc_auc_score(y_train_bin[:, i], y_train_pred_proba[:, i])

print("\nTraining Results")
print(f"Training Accuracy: {train_accuracy}")
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("Training Classification Report:")
print(train_class_report)
print(f"Training AUC: {train_roc_auc}")

# Evaluate the model on the validation set
y_val_pred = best_rf_model.predict(X_val)
y_val_pred_proba = best_rf_model.predict_proba(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
val_class_report = classification_report(y_val, y_val_pred)

# Compute ROC AUC scores for each class
y_val_bin = label_binarize(y_val, classes=[0, 1, 2])
val_roc_auc = {}
for i in range(len(np.unique(y))):
    val_roc_auc[i] = roc_auc_score(y_val_bin[:, i], y_val_pred_proba[:, i])

print("\nValidation Results")
print(f"Validation Accuracy: {val_accuracy}")
print("Validation Confusion Matrix:")
print(val_conf_matrix)
print("Validation Classification Report:")
print(val_class_report)
print(f"Validation AUC: {val_roc_auc}")

# Evaluate the model on the test set
y_test_pred = best_rf_model.predict(X_test)
y_test_pred_proba = best_rf_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
test_class_report = classification_report(y_test, y_test_pred)

# Compute ROC AUC scores for each class
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
test_roc_auc = {}
for i in range(len(np.unique(y))):
    test_roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_test_pred_proba[:, i])

print("\nTest Results")
print(f"Test Accuracy: {test_accuracy}")
print("Test Confusion Matrix:")
print(test_conf_matrix)
print("Test Classification Report:")
print(test_class_report)
print(f"Test AUC: {test_roc_auc}")

# Plot ROC Curve for test data
fpr = {}
tpr = {}
roc_auc = {}
n_classes = len(np.unique(y))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_test_pred_proba[:, i])

plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot the feature importances
importances = best_rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf_model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("\nFeature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. feature {indices[f]} ({importances[indices[f]]})")

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()





print(
      """ ### XGBoost Classifier ## """
)

# Calculate scale_pos_weight
pos_count = np.sum(y_train == 1)
neg_count = np.sum(y_train == 0)
scale_pos_weight = neg_count / pos_count

# XGBoost classifier
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Parameter tunning
param_grid = {
    'eta': [0.1],
    'gamma': [0.1],
    'max_depth': [5],
    'min_child_weight': [3],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'lambda': [1],
    'alpha': [0],
    'scale_pos_weight': [scale_pos_weight],
    'n_estimators': [100],
    'learning_rate': [0.1]
}

# Initialize GridSearchCV with StratifiedKFold
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='roc_auc_ovr')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found by grid search
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the model on the training set
y_train_pred = best_model.predict(X_train)
y_train_pred_proba = best_model.predict_proba(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
train_class_report = classification_report(y_train, y_train_pred)
train_roc_auc = roc_auc_score(label_binarize(y_train, classes=[0, 1, 2]), y_train_pred_proba, multi_class='ovr')

print("Training Results")
print(f"Training Accuracy: {train_accuracy}")
print("Training Confusion Matrix:")
print(train_conf_matrix)
print("Training Classification Report:")
print(train_class_report)
print(f"Training AUC: {train_roc_auc}")

# Validate the model
y_val_pred = best_model.predict(X_val)
y_val_pred_proba = best_model.predict_proba(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)
val_class_report = classification_report(y_val, y_val_pred)
val_roc_auc = roc_auc_score(label_binarize(y_val, classes=[0, 1, 2]), y_val_pred_proba, multi_class='ovr')

print("Validation Results")
print(f"Validation Accuracy: {val_accuracy}")
print("Validation Confusion Matrix:")
print(val_conf_matrix)
print("Validation Classification Report:")
print(val_class_report)
print(f"Validation AUC: {val_roc_auc}")

# Test the model
y_test_pred = best_model.predict(X_test)
y_test_pred_proba = best_model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)
test_class_report = classification_report(y_test, y_test_pred)
test_roc_auc = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2]), y_test_pred_proba, multi_class='ovr')

print("Test Results")
print(f"Test Accuracy: {test_accuracy}")
print("Test Confusion Matrix:")
print(test_conf_matrix)
print("Test Classification Report:")
print(test_class_report)
print(f"Test AUC: {test_roc_auc}")

# Plot ROC Curve for test data
fpr = {}
tpr = {}
roc_auc = {}
n_classes = len(np.unique(y))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_test_pred_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_test_pred_proba[:, i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot feature importance
feature_importances = pd.DataFrame(best_model.feature_importances_,
                                   index=df.drop('Diabetes_012', axis=1).columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importances')
plt.show()




print(
      """ ### K-Nearest Neighbors Classifier ## """
)

# Initialize the KNN classifier
clf = KNeighborsClassifier()

# Define a smaller parameter grid for faster execution, using only Euclidean distance
param_grid = {
    'n_neighbors': [50],
    'weights': ['uniform', 'distance']
}

# Initialize GridSearchCV with fewer folds for faster execution
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

# Train the classifier on the balanced data
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best estimator
best_clf = grid_search.best_estimator_

# Evaluate the model on the training set
y_train_pred = best_clf.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Print training classification report
print('Training Classification Report:')
print(classification_report(y_train_resampled, y_train_pred))

# Print training confusion matrix
print('Training Confusion Matrix:')
print(confusion_matrix(y_train_resampled, y_train_pred))

# Evaluate the model on the validation set
y_val_pred = best_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Print validation classification report
print('Validation Classification Report:')
print(classification_report(y_val, y_val_pred))

# Print validation confusion matrix
print('Validation Confusion Matrix:')
print(confusion_matrix(y_val, y_val_pred))

# Evaluate the model on the test set using the best estimator
y_test_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Print classification report
print('Test Classification Report:')
print(classification_report(y_test, y_test_pred))

# Print confusion matrix
print('Test Confusion Matrix:')
print(confusion_matrix(y_test, y_test_pred))

# Print count and percentage of each prediction
prediction_counts = pd.Series(y_test_pred).value_counts()
prediction_percentages = pd.Series(y_test_pred).value_counts(normalize=True) * 100

print('\nPredicted Value Counts:')
print(prediction_counts)

print('\nPredicted Value Percentages:')
print(prediction_percentages)

# Print the best parameters
print(f'Best Parameters: {grid_search.best_params_}')

# Calculate probabilities for ROC curve
y_test_prob = best_clf.predict_proba(X_test)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_test, y_test_prob)
print(f'AUC-ROC Score: {roc_auc:.2f}')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
