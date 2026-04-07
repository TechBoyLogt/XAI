import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import shap
import dice_ml
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Load dataset
df = pd.read_csv("diabetes.csv")
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale (optional for tree models, but kept for uniformity)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)

    print(f" Accuracy: {accuracy:.3f}")
    print(f" Sensitivity (Recall): {sensitivity:.3f}")
    print(f" Specificity: {specificity:.3f}")
    return accuracy, sensitivity, specificity

# Train Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)

# Predict and Evaluate
y_pred_lr = model_lr.predict(X_test_scaled)
evaluate_model(y_test, y_pred_lr)

# SHAP
explainer_lr = shap.Explainer(model_lr, X_train_scaled)
shap_values_lr = explainer_lr(X_test_scaled)

# SHAP Visualizations
sample_index = 0
shap.plots.waterfall(shap_values_lr[sample_index])
shap.plots.bar(shap_values_lr)
shap.plots.beeswarm(shap_values_lr)
shap.plots.force(shap_values_lr[0])
