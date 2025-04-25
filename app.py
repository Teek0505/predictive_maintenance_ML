import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("xgb_model.pkl")

st.title("Predictive Maintenance App")
st.write("Enter sensor data to predict machine failure:")

# Input fields
air_temp = st.number_input("Air Temperature (K)", value=300.0)
process_temp = st.number_input("Process Temperature (K)", value=310.0)
rot_speed = st.number_input("Rotational Speed (rpm)", value=1500.0)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=100.0)

# Predict button
if st.button("Predict Failure"):
    input_data = pd.DataFrame([[air_temp, process_temp, rot_speed, torque, tool_wear]],
                              columns=['Air temperature [K]', 'Process temperature [K]',
                                       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'])
    prediction = model.predict(input_data)
    result = "⚠️ Machine Failure Likely!" if prediction[0] == 1 else "✅ No Failure Predicted."
    st.subheader(result)
