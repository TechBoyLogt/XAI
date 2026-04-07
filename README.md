# Diabetes Risk Prediction with Explainable AI

This project demonstrates how to explain machine learning predictions for diabetes risk using two explainability methods:

- SHAP (feature contribution explanations)
- DiCE (counterfactual explanations)

The app is built with Streamlit and uses the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
## Project Overview

The dashboard trains a Logistic Regression model on the dataset, predicts diabetes risk for a selected patient, and then explains the prediction using:

- SHAP waterfall and bar plots to show feature impact
- DiCE counterfactual examples to show what feature changes could flip a high-risk prediction to low-risk

## Files in This Repository

- `streamlit_app.py`: Main Streamlit dashboard
- `XAIfinalshap.py`: Standalone SHAP experimentation script
- `XAIfinaldice.py`: Standalone DiCE experimentation script
- `diabetes.csv`: Dataset used for training/testing
- `Dashboard photos/`: Optional screenshots/images

## Requirements

Install dependencies with:

```bash
pip install streamlit pandas numpy scikit-learn shap dice-ml plotly matplotlib xgboost
```

## How to Run the Dashboard

From the project folder, run:

```bash
python -m streamlit run streamlit_app.py
```

Then open the local URL shown in your terminal, which is usually http://localhost:8501.

## What the Dashboard Shows

1. Patient-level prediction (high/low risk)
2. Predicted diabetes probability
3. Actual label from the test set
4. SHAP explanation plots
5. DiCE counterfactual examples (for high-risk predictions)

## Notes

- Features are standardized before model training.
- Counterfactuals are generated only when the patient is predicted as high risk.
- This project is for explainability and educational purposes; it is not a clinical decision tool.

## Clone the Repository

```bash
git clone https://github.com/TechBoyLogt/XAI.git
cd XAI
```
