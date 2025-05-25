import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("notebooks/models/launch_success.pkl")

# Set page config
st.set_page_config(page_title="🚀 SpaceX Launch Success Predictor", layout="centered")
st.title("🚀 SpaceX Launch Success Predictor")

st.markdown("""
Enter all launch parameters to predict whether the SpaceX launch will be successful.
""")

# Input fields
rocket_encoded = st.selectbox("🚀 Rocket (Encoded)", options=[0, 1, 2], format_func=lambda x: f"Rocket {x}")
launchpad_encoded = st.selectbox("🛰️ Launchpad (Encoded)", options=[0, 1, 2], format_func=lambda x: f"Launchpad {x}")
payload_mass = st.number_input("📦 Payload Mass (kg)", min_value=0.0, max_value=50000.0, value=6000.0)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=-100.0, max_value=100.0, value=25.0)
humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
wind_speed = st.number_input("💨 Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0)
year = st.number_input("📅 Year", min_value=2002, max_value=2030, value=2020)
month = st.number_input("📆 Month", min_value=1, max_value=12, value=6)
day = st.number_input("📅 Day", min_value=1, max_value=31, value=15)
hour = st.number_input("⏰ Hour", min_value=0, max_value=23, value=13)

# Predict
if st.button("Predict Launch Success"):
    input_data = np.array([[rocket_encoded, launchpad_encoded, payload_mass,
                            temperature, humidity, wind_speed,
                            year, month, day, hour]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Predicted: **Successful Launch** with {probability * 100:.2f}% confidence.")
    else:
        st.error(f"❌ Predicted: **Failed Launch** with {(1 - probability) * 100:.2f}% confidence.")

# Footer
st.markdown("---")
st.markdown("Developed by Zafir Abdullah | Powered by Machine Learning")
