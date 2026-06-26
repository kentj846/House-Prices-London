import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="London House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 London House Price Predictor")
st.write("Input a property's details below to estimate its current valuation based on historical trends.")

# 2. Helper to load the model (add joblib.dump(model, 'london_model.pkl') in your notebook first!)
@st.cache_resource
def load_model():
    try:
        return joblib.load("london_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'london_model.pkl' not found. Please train and save your model first.")
        return None

model = load_model()

if model is not None:
    st.subheader("Property Characteristics")
    
    # 3. Create User Inputs in Sidebar/Columns
    col1, col2 = st.columns(2)
    
    with col1:
        floor_area = st.number_input("Floor Area (Sq Meters)", min_value=15, max_value=1000, value=75, step=5)
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=2)
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
        
    with col2:
        latitude = st.number_input("Latitude", min_value=51.25, max_value=51.70, value=51.5074, format="%.4f")
        longitude = st.number_input("Longitude", min_value=-0.55, max_value=0.30, value=-0.1278, format="%.4f")
        
        # Target Encoded Outcode Input
        # (In production, map a dropdown selection to its target-encoded mean value)
        outcode_encoded = st.number_input("Outcode Average Price Benchmark (£)", min_value=100000, max_value=5000000, value=550000)

    # Property Type One-Hot Encoding Toggles
    st.write("---")
    st.subheader("Property Type")
    prop_type = st.selectbox("Select Property Type", ["Flat", "Terraced", "Semi-Detached", "Detached"])
    
    # Map the dropdown to the exact One-Hot structural columns your model expects
    is_flat = 1 if prop_type == "Flat" else 0
    is_terraced = 1 if prop_type == "Terraced" else 0
    is_semi = 1 if prop_type == "Semi-Detached" else 0
    
    # 4. Predict Button
    if st.button("Calculate Estimated Value", type="primary"):
        # Match the exact column structure of your X_train matrix!
        # Update this list to match your exact engineered columns
        input_data = pd.DataFrame([{
            'floorAreaSqM': floor_area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'latitude': latitude,
            'longitude': longitude,
            'outcode_encoded': outcode_encoded,
            'propertyType_Flat': is_flat,
            'propertyType_Semi-Detached': is_semi,
            'propertyType_Terraced': is_terraced
        }])
        
        # Generate prediction
        prediction = model.predict(input_data)[0]
        
        # Reverse log transformation if you used it during training:
        # prediction = np.expm1(prediction)
        
        # Display the result
        st.success(f"### Estimated Market Price: **£{prediction:,.2f}**")