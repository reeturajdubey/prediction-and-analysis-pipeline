import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load dataset for feature names and default values
data = load_breast_cancer()
feature_names = data.feature_names
X = pd.DataFrame(data.data, columns=feature_names)

# Page Conf
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a breast mass is benign or malignant using various Machine Learning models.")

# Sidebar - Model Selection
st.sidebar.header("Model Selection")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]
selected_model_name = st.sidebar.selectbox("Choose a Classification Model", model_options)

# Load Model and Scaler
@st.cache_resource
def load_model_and_scaler(model_name):
    filename = f"model/{model_name.replace(' ', '_').lower()}.pkl"
    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Model file {filename} not found. Please run train_model.py first.")
        return None, None

model, scaler = load_model_and_scaler(selected_model_name)

# Input Features
st.sidebar.header("Input Features")
st.sidebar.write("Adjust the values below:")

input_data = {}
# Using mean values as defaults
defaults = X.mean()

# Create input fields (using sliders for better UX, centered around mean)
for feature in feature_names:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    default_val = float(defaults[feature])
    
    input_data[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=(max_val - min_val) / 100
    )

# Main Area - Prediction
if st.button("Predict"):
    if model and scaler:
        # Prepare input
        input_df = pd.DataFrame([input_data])
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None
        
        # Display Result
        st.subheader("Prediction Result")
        result_text = "Malignant" if prediction == 0 else "Benign" # 0 is malignant in sklearn dataset usually? Wait, let's verify.
        # Sklearn breast cancer: 0 = Malignant, 1 = Benign. 
        # Actually usually 0 is negative class (Benign) and 1 is positive (Malignant)? 
        # Let's check documentation or dataset description.
        # In sklearn load_breast_cancer():
        # target_names: array(['malignant', 'benign'], dtype='<U9')
        # So 0 is malignant, 1 is benign.
        
        color = "red" if prediction == 0 else "green"
        st.markdown(f"<h3 style='color: {color};'>{result_text}</h3>", unsafe_allow_html=True)
        
        if prediction_proba is not None:
             st.write(f"Confidence (Benign): {prediction_proba[1]:.2f}")
             st.write(f"Confidence (Malignant): {prediction_proba[0]:.2f}")

        # Dataset Info
        with st.expander("Dataset Information"):
            st.write(data.DESCR)
    else:
        st.error("Model could not be loaded.")

# Load and display metrics
st.markdown("---")
st.subheader("Model Performance Comparison")
try:
    metrics_df = pd.read_csv("model_metrics.csv")
    st.dataframe(metrics_df)
    
    # Highlight selected model
    st.write(f"**Current Model ({selected_model_name}) Metrics:**")
    st.table(metrics_df[metrics_df["ML Model Name"] == selected_model_name])
except FileNotFoundError:
    st.warning("Metrics file not found. Run train_model.py first.")
