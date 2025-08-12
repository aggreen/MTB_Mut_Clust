import pandas as pd
import numpy as np
import re
from evcouplings.compare import DistanceMap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score,roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay)
from scipy import stats
import os
def data_split(train_data,numeric_columns):

    filtered_indices = train_data.index
    X = train_data[numeric_columns].values
    y = train_data['binary_confidence'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X_scaled, y, filtered_indices, test_size=0.3, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def model_training(X_train, y_train,X_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)
    y_prob_test = rf.predict_proba(X_test)
    return rf, y_pred_test, y_prob_test

def performance_evaluation(y_test, y_pred,y_prob):
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    precision = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['precision'] #dataset is class imbalanced
    accuracy= accuracy_score(y_test, y_pred)

    # Create mapping from class label to index
    class_to_index = {cls: idx for idx, cls in enumerate(np.unique(y_test))}

    fpr, tpr, roc_auc = {}, {}, {}
    for cls in class_to_index:
        idx = class_to_index[cls]
        fpr[cls], tpr[cls], _ = roc_curve(y_test == cls, y_prob[:, idx])
        roc_auc[cls] = auc(fpr[cls], tpr[cls])
        
    # #plot roc-auc curve     
    # plt.figure(figsize=(8, 6))
    # for cls in class_to_index:
    #     plt.plot(fpr[cls], tpr[cls], label=f"Class {cls} (AUC = {roc_auc[cls]:.2f})")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend()
    # plt.grid()
    # plt.show()
    return accuracy, roc_auc, precision

def plot_feature_importance(model,numeric_columns):
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(numeric_columns)), feature_importances[indices], align="center")
    plt.xticks(range(len(numeric_columns)), [numeric_columns[i] for i in indices], rotation=45)
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()
    return feature_importances


def predict_category_3(category_3_data, model,original_df,numeric_columns,y_test,y_pred,test_indices):
    category_3_data[numeric_columns] = category_3_data[numeric_columns].fillna(0)
    category_3_preds = model.predict(category_3_data[numeric_columns])

    category_3_data['predicted_confidence'] = category_3_preds
    original_df.loc[original_df['confidence'] == "3) Uncertain significance", 'predicted_confidence'] = category_3_preds
    
    # Save prediction results
    original_df.to_csv(f"../data/predicted_confidence_{numeric_columns}.csv", index=False)



    predicted_confidence_counts = category_3_data['predicted_confidence'].value_counts()
    confidence_0_count = predicted_confidence_counts.get(0.0, 0)  # Get count of 0.0 predictions, default to 0 if missing
    confidence_1_count = predicted_confidence_counts.get(1.0, 0)  # Get count of 1.0 predictions, default to 0 if missing
    # Calculate total predictions for category 3 data
    total_predictions = len(category_3_data)

    # Calculate percentages
    confidence_0_percentage = (confidence_0_count / total_predictions) * 100
    confidence_1_percentage = (confidence_1_count / total_predictions) * 100

    # Identify misclassified indices and their rows
    misclassified_indices = np.where(y_test != y_pred)[0]
    misclassified_original_indices = test_indices[misclassified_indices]

    # Extract misclassified samples
    misclassified_samples = original_df.loc[misclassified_original_indices].copy()

    # Add predicted confidence (predicted labels) to misclassified samples
    misclassified_samples['predicted_confidence'] = y_pred[misclassified_indices]
    
    return misclassified_original_indices, misclassified_samples, confidence_0_count, confidence_0_percentage, confidence_1_count, confidence_1_percentage

