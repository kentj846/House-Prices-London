import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="London Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 London House Price Predictor")
st.write("Input the property dimensions and location metrics to calculate an estimated market valuation.")

# Safe Model Loader
@st.cache_resource
def load_model():
    try:
        return joblib.load("london_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'london_model.pkl' not found. Ensure it is saved in this exact folder.")
        return None

model = load_model()

if model is not None:
    st.subheader("Property Specifications")
    
    # Structural Features
    floor_area = st.number_input("Floor Area (Square Metres)", min_value=15, max_value=1000, value=75, step=5)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=2)
    with col2:
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=1)
    with col3:
        living_rooms = st.slider("Living Rooms", min_value=0, max_value=5, value=1)
        
    st.write("---")
    st.subheader("Location & Value Benchmarks")
    
    col4, col5 = st.columns(2)
    with col4:
        distance_to_centre = st.slider(
            "Distance to Central London (km)", 
            min_value=0.0, 
            max_value=50.0, 
            value=10.0, 
            step=0.1
        )
    with col5:
        outcode_encoded = st.number_input(
            "Postcode Average Benchmark Price (£)", 
            min_value=100000, 
            max_value=5000000, 
            value=550000,
            step=25000
        )

    st.write("---")
    st.subheader("Property Type")
    
    # Clean list matching all your one-hot encoded suffix categories
    property_types = [
        "Converted Flat", "Detached Bungalow", "Detached House", "Detached Property",
        "End Terrace Bungalow", "End Terrace House", "End Terrace Property", "Flat/Maisonette",
        "Mid Terrace Bungalow", "Mid Terrace House", "Mid Terrace Property", "Purpose Built Flat",
        "Semi-Detached Bungalow", "Semi-Detached House", "Semi-Detached Property",
        "Terrace Property", "Terraced", "Terraced Bungalow", "Other Type"
    ]
    
    selected_type = st.selectbox("Select Property Type", property_types)

    st.write("---")
    
    # Predict Button
    if st.button("Calculate Valuation", type="primary"):
        
        # 1. Start with a baseline dictionary of the numerical fields (Without sale estimate)
        input_dict = {
            'bathrooms': float(bathrooms),
            'bedrooms': float(bedrooms),
            'floorAreaSqM': float(floor_area),
            'livingRooms': float(living_rooms),
            'distance_to_centre': float(distance_to_centre),
            'outcode_encoded': float(outcode_encoded)
        }
        
        # 2. Add all 18 propertyType one-hot encoded flags (set them to 0 by default)
        all_one_hot_cols = [
            "propertyType_Converted Flat", "propertyType_Detached Bungalow", "propertyType_Detached House", 
            "propertyType_Detached Property", "propertyType_End Terrace Bungalow", "propertyType_End Terrace House", 
            "propertyType_End Terrace Property", "propertyType_Flat/Maisonette", "propertyType_Mid Terrace Bungalow", 
            "propertyType_Mid Terrace House", "propertyType_Mid Terrace Property", "propertyType_Purpose Built Flat", 
            "propertyType_Semi-Detached Bungalow", "propertyType_Semi-Detached House", "propertyType_Semi-Detached Property", 
            "propertyType_Terrace Property", "propertyType_Terraced", "propertyType_Terraced Bungalow"
        ]
        
        for col in all_one_hot_cols:
            input_dict[col] = 0
            
        # 3. Set the one column the user actually selected to 1
        target_col = f"propertyType_{selected_type}"
        if target_col in input_dict:
            input_dict[target_col] = 1
            
        # 4. Convert dictionary into a DataFrame
        input_data = pd.DataFrame([input_dict])
        
        # 5. ENFORCE EXACT COLUMN ORDER MATCHING YOUR 24 TRAINING FEATURES
        exact_column_order = [
            'bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
            'distance_to_centre', 'propertyType_Converted Flat', 'propertyType_Detached Bungalow',
            'propertyType_Detached House', 'propertyType_Detached Property', 'propertyType_End Terrace Bungalow',
            'propertyType_End Terrace House', 'propertyType_End Terrace Property', 'propertyType_Flat/Maisonette',
            'propertyType_Mid Terrace Bungalow', 'propertyType_Mid Terrace House', 'propertyType_Mid Terrace Property',
            'propertyType_Purpose Built Flat', 'propertyType_Semi-Detached Bungalow', 'propertyType_Semi-Detached House',
            'propertyType_Semi-Detached Property', 'propertyType_Terrace Property', 'propertyType_Terraced',
            'propertyType_Terraced Bungalow', 'outcode_encoded'
        ]
        input_data = input_data[exact_column_order]
        
        # Generate raw prediction
        prediction = model.predict(input_data)
        
        # If your notebook used log transformation on the price, uncomment the next line:
        # prediction = np.expm1(prediction)
        
        # Display formatted output
        st.success(f"### Estimated Market Price: **£{prediction[0]:,.2f}**")