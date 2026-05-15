import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load('car_price_model.joblib')

scaler = joblib.load('scaler.joblib')

columns = joblib.load('model_columns.joblib')

# App title
st.title("Car Price Prediction App")

# User Inputs
fuel_type = st.selectbox(
    "Fuel Type",
    ['Petrol', 'Diesel']
)

gear_type = st.selectbox(
    "Gear Type",
    ['Automatic', 'Manual']
)

make = st.text_input("Car Make")

model_name = st.text_input("Car Model")

year = st.number_input(
    "Year of Manufacture",
    min_value=1990,
    max_value=2026
)

condition = st.selectbox(
    "Condition",
    ['Nigerian Used', 'Foreign Used', 'Brand New']
)

mileage = st.number_input("Mileage")

engine_size = st.number_input("Engine Size")

selling_condition = st.text_input("Selling Condition")

bought_condition = st.text_input("Bought Condition")

# Prediction button
if st.button("Predict Price"):

    # Create dataframe
    input_data = pd.DataFrame([{
        'fuel type': fuel_type,
        'gear type': gear_type,
        'Make': make,
        'Model': model_name,
        'Year of manufacture': year,
        'Condition': condition,
        'Mileage': mileage,
        'Engine Size': engine_size,
        'Selling Condition': selling_condition,
        'Bought Condition': bought_condition
    }])

    # Encode categorical variables
    input_data = pd.get_dummies(input_data)

    # Match training columns
    input_data = input_data.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Show result
    st.success(
        f"Estimated Car Price: ₦{prediction[0]:,.0f}"
    )
