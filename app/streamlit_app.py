"""
Flexible Nairobi Property Price Predictor
Run with: streamlit run app/flexible_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Nairobi Property Predictor", page_icon="🏠")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/features_listings.csv')

# Load best model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/best_model.pkl')
        features = joblib.load('models/best_features.pkl')
        metadata = pd.read_csv('models/best_model_metadata.csv', index_col=0).squeeze()
        encoders = joblib.load('models/best_encoders.pkl') if os.path.exists('models/best_encoders.pkl') else None
        return model, features, metadata, encoders
    except:
        return None, None, None, None

import os
df = load_data()
model, features, metadata, encoders = load_model()

st.title("🏠 Nairobi Property Price Predictor")

if model is None:
    st.error("No trained model found. Please run the training script first:")
    st.code("python scripts/04_model_training_flexible.py")
    st.stop()

st.info(f"**Model used:** {metadata.get('model_name', 'Unknown')}")
st.info(f"**Accuracy:** ± KES {float(metadata['mae']):,.0f}")

# Create input form
st.subheader("Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", sorted(df['location'].dropna().unique()))
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

with col2:
    prop_type = st.selectbox("Property Type", sorted(df['property_type'].dropna().unique()))
    if 'size_sqm' in str(features):
        size = st.number_input("Size (sqm)", min_value=20, max_value=1000, value=100)

if st.button("Predict Price", type="primary"):
    # Prepare input based on available features
    input_data = {}
    
    for feat in features:
        if 'location' in feat:
            input_data[feat] = 0  # Will be encoded
        elif 'property' in feat:
            input_data[feat] = 0  # Will be encoded
        elif 'bedrooms' in feat:
            input_data[feat] = bedrooms
        elif 'size_sqm' in feat:
            input_data[feat] = size
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    pred_log = model.predict(input_df)[0]
    prediction = np.exp(pred_log)
    
    # Display result
    st.markdown("---")
    st.subheader("Predicted Price")
    st.markdown(f"# KES {prediction:,.0f}")
    st.markdown(f"**Range:** KES {prediction - float(metadata['mae']):,.0f} - KES {prediction + float(metadata['mae']):,.0f}")