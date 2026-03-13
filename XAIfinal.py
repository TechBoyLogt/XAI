!pip install streamlit pandas plotly shap matplotlib ucimlrepo scikit-learn
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_resource
def setup_experiment():
    repo = fetch_ucirepo(id=144)
    
    X = repo.data.features.select_dtypes(include=['number']) 
    
    y = repo.data.targets.iloc[:, 0].map({1: 0, 2: 1})
    
    feature_mapping = {
        'Attribute2': 'Duration (months)',
        'Attribute5': 'Credit Amount',
        'Attribute8': 'Installment Rate (%)',
        'Attribute11': 'Residence Duration (years)',
        'Attribute13': 'Age (years)',
        'Attribute16': 'Existing Credits',
        'Attribute18': 'Number of Dependents'
    }
    X = X.rename(columns=feature_mapping)
    
    # Train test split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training the random forest model Model
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    return X, X_test, model, explainer

X, X_test, model, explainer = setup_experiment()
# This is the user interface made with streamlit which is a framework which turns pythoncode into a frontend.
st.set_page_config(page_title="XAI Trust Study", layout="wide")
st.title("Credit Risk Transparency Experiment")

"""
This dashboard is made to show the difference between a statical explanation 
and a interactive explanation of shap values. The information is fetched from the UCIML repo.
With the German Statlog credit dataset being used in conjunction with a random forest classifier.
"""

# This allows the user to control the interactive part of the shap explanations
st.sidebar.markdown("### Participant Controls")
user_idx = st.sidebar.number_input("Select Applicant ID", 0, len(X_test)-1, 10)

# Fetches the data of the current participant
instance = X_test.iloc[user_idx:user_idx+1]
prediction_prob = model.predict_proba(instance)[0][1]

#Statistical explanations usingplotly
st.header("1. Statistical Explanation ")
st.write("This chart shows the general rules and weights the bank uses for everyone.")


importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)

fig_global = px.bar(importances, x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Viridis',
                    text_auto='.3f') 

fig_global.update_layout(title="Top Factors Influencing All Bank Decisions")
st.plotly_chart(fig_global, width='stretch')

st.divider()

# Local SHAP explanations
st.header("2. SHAP Explanation")
st.write(f"The model predicts a **{prediction_prob:.2%}** risk of default for Applicant #{user_idx}.  The higher(+/-) the shap value the more it influences the decision of the bank. ")


shap_obj = explainer(instance)

if len(shap_obj.values.shape) == 3:
    sv = shap_obj.values[0, :, 1].flatten().tolist()
    base_value = float(shap_obj.base_values[0, 1])
else:
    sv = shap_obj.values[0].flatten().tolist()
    base_value = float(shap_obj.base_values[0])

feature_names = X.columns.tolist()

text_labels = [f"{val:+.3f}" for val in sv]

# Creating a waterfall plot for to visualize the interactive data
fig_shap = go.Figure(go.Waterfall(
    orientation = "h",
    measure = ["relative"] * len(sv),
    y = feature_names,
    x = sv, 
    text = text_labels,          
    textposition = "outside",    
    base = base_value,
    connector = {"line":{"color":"rgb(63, 63, 63)"}},
    increasing = {"marker":{"color":"#ef553b"}}, 
    decreasing = {"marker":{"color":"#636efa"}}  
))

fig_shap.update_layout(
    title="How your specific features moved the risk score from the average",
    waterfallgap=0.4,
    margin=dict(l=150) 
)
st.plotly_chart(fig_shap, width= "stretch")


st.divider()

# Shap global statistical explanations showing the distribution for every single participant in the database.
st.header("3. SHAP Summary Full Dataset")
st.write("This chart shows how different features impact the risk score across **all** applicants.")

@st.cache_data
def get_global_shap_values(_explainer, X_data):
    shap_vals = _explainer(X_data)
    
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]
        
    return shap_vals

shap_values_all = get_global_shap_values(explainer, X_test)

fig_summary = plt.figure(figsize=(7, 6))

shap.summary_plot(shap_values_all, X_test, show=False)

st.pyplot(plt.gcf())

plt.clf()