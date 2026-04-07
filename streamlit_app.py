import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import dice_ml
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


st.set_page_config(page_title="Diabetes XAI Dashboard", layout="wide")


st.title("Diabetes Risk Prediction with Explainable AI")
st.markdown("This dashboard was created to show the difference between SHAP and DiCE explanations for a Logistic Regression model trained on the Pima Indians Diabetes dataset.")


# Cache the model and data, this is done to make sure that model runs smoothly
@st.cache_resource
def load_model_and_data():
    # loading the dataset
    df = pd.read_csv("diabetes.csv")
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    # traintest split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # training model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # shap explainer
    explainer = shap.Explainer(model, X_train_scaled)

    # dice explainer
    data_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)
    dice_data = dice_ml.Data(dataframe=data_df, continuous_features=X_train_scaled.columns.tolist(), outcome_name='Outcome')
    dice_model = dice_ml.Model(model=model, backend='sklearn')
    dice_exp = dice_ml.Dice(dice_data, dice_model, method='random')

    return model, X_test, X_test_scaled, y_test, explainer, dice_exp, scaler

# loading model and data
model, X_test, X_test_scaled, y_test, explainer, dice_exp, scaler = load_model_and_data()

# Sidebar fot the interface
st.sidebar.header("Patient Selection")
patient_idx = st.sidebar.slider("Select Patient Index", 0, len(X_test)-1, 0)

# Loading patient data
patient_raw = X_test.iloc[patient_idx]
patient_scaled = X_test_scaled.iloc[patient_idx:patient_idx+1]

# Prediction
prediction = model.predict(patient_scaled)[0]
prob = model.predict_proba(patient_scaled)[0][1]

# Creating 3 width columns to show prediction, probability and outcome
column1, column2, column3 = st.columns(3)
with column1:
    risk = "🔴 High Risk for Diabetes" if prediction == 1 else "🟢 Low Risk for Diabetes"
    st.metric("Prediction", risk)
with column2:
    st.metric("Diabetes Probability", f"{prob:.1%}")
with column3:
    st.metric("Actual Outcome", "Diabetic" if y_test.iloc[patient_idx] == 1 else "Non-Diabetic")

st.subheader("Patient Features")
st.dataframe(patient_raw.to_frame().T)

# SHAP Explanations
st.header("SHAP Feature Importance")
shap_values = explainer(patient_scaled)

# SHAP Waterfall Plot
st.subheader("SHAP Waterfall Plot")
fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)

# SHAP Bar Plot
st.subheader("SHAP Feature Importance")
fig2, ax2 = plt.subplots(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
st.pyplot(fig2)

# DiCE Counterfactuals
st.header("DiCE Counterfactual Explanations")
if prediction == 1:
    st.write("Generating counterfactual scenarios that would change the prediction to Low Risk")
    cf = dice_exp.generate_counterfactuals(patient_scaled, total_CFs=3, desired_class="opposite")
    if cf.cf_examples_list and len(cf.cf_examples_list) > 0:
        cf_df = cf.cf_examples_list[0].final_cfs_df
        if cf_df is not None and len(cf_df) > 0:
            st.subheader("Counterfactual Explanations Table")
            st.dataframe(cf_df)
            
            # Plotly comparison plot
            st.subheader("Original vs Counterfactual Comparison")
            original = patient_scaled.iloc[0]
            cf_values = cf_df.iloc[0]
            features = list(X_test.columns)
            
            fig_dice = go.Figure()
            fig_dice.add_trace(go.Bar(x=features, y=original, name='Original', marker_color='blue'))
            fig_dice.add_trace(go.Bar(x=features, y=cf_values, name='Counterfactual', marker_color='red'))
            fig_dice.update_layout(
                title='Feature Comparison',
                xaxis_title='Features',
                yaxis_title='Scaled Values',
                barmode='group'
            )
            st.plotly_chart(fig_dice)
        else:
            st.write("No counterfactuals could be generated for this patient.")
    else:
        st.write("No counterfactual examples were generated.")
else:
    st.success("This patient is already predicted as Low Risk. No counterfactuals needed!")

# Footer
st.markdown("---")
st.markdown("Created by Rimski Logtens | Dataset: Pima Indians Diabetes Database | Model: Logistic Regression")