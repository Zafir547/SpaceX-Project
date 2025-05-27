# import streamlit as st
# import joblib
# import numpy as np

# # Load trained model
# model = joblib.load("notebooks/models/launch_success.pkl")

# # Set page config
# st.set_page_config(page_title="🚀 SpaceX Launch Success Predictor", layout="centered")
# st.title("🚀 SpaceX Launch Success Predictor")

# st.markdown("""
# Enter all launch parameters to predict whether the SpaceX launch will be successful.
# """)

# # Input fields
# rocket_encoded = st.selectbox("🚀 Rocket (Encoded)", options=[0, 1, 2], format_func=lambda x: f"Rocket {x}")
# launchpad_encoded = st.selectbox("🛰️ Launchpad (Encoded)", options=[0, 1, 2], format_func=lambda x: f"Launchpad {x}")
# payload_mass = st.number_input("📦 Payload Mass (kg)", min_value=0.0, max_value=50000.0, value=6000.0)
# temperature = st.number_input("🌡️ Temperature (°C)", min_value=-100.0, max_value=100.0, value=25.0)
# humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
# wind_speed = st.number_input("💨 Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0)
# year = st.number_input("📅 Year", min_value=2002, max_value=2030, value=2020)
# month = st.number_input("📆 Month", min_value=1, max_value=12, value=6)
# day = st.number_input("📅 Day", min_value=1, max_value=31, value=15)
# hour = st.number_input("⏰ Hour", min_value=0, max_value=23, value=13)

# # Predict
# if st.button("Predict Launch Success"):
#     input_data = np.array([[rocket_encoded, launchpad_encoded, payload_mass,
#                             temperature, humidity, wind_speed,
#                             year, month, day, hour]])
#     prediction = model.predict(input_data)[0]
#     probability = model.predict_proba(input_data)[0][1]

#     if prediction == 1:
#         st.success(f"✅ Predicted: **Successful Launch** with {probability * 100:.2f}% confidence.")
#     else:
#         st.error(f"❌ Predicted: **Failed Launch** with {(1 - probability) * 100:.2f}% confidence.")

# # Footer
# st.markdown("---")
# st.markdown("Developed by Zafir Abdullah | Powered by Machine Learning")

# Code 2

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

st.set_page_config(page_title="🚀 SpaceX Launch Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("processed_spacex_data2.csv")
    return df

df = load_data()

# Load model
model = joblib.load("notebooks/models/launch_success.pkl")

# App config

st.title("🚀 SpaceX Launch Analysis & Success Prediction")

st.sidebar.header("🔎 Filter Launch Data")
year_filter = st.sidebar.multiselect("Select Year(s):", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
site_filter = st.sidebar.multiselect("Select Launch Site(s):", df['launchpad_encoded'].unique(), default=df['launchpad_encoded'].unique())

filtered_df = df[(df['year'].isin(year_filter)) & (df['launchpad_encoded'].isin(site_filter))]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Historical Analysis", "🗺️ Launch Map", "🤖 Predict Launch Success"])

# ========== TAB 1 ==========
with tab1:
    st.subheader("📊 Historical Launch Data Overview")

    st.dataframe(filtered_df[['name', 'date_utc', 'rocket_encoded', 'launchpad_encoded',
                              'payload_mass', 'lon', 'lat', 'temperature', 'humidity', 'wind_speed',
                              'year', 'month', 'day', 'hour', 'success']], use_container_width=True)

    st.markdown("### 📈 Success Rate by Launch Site")
    success_rate = filtered_df.groupby('launchpad_encoded')['success'].mean().reset_index()
    st.bar_chart(success_rate.set_index('launchpad_encoded'))

    st.markdown("### 🪂 Payload Mass vs Launch Success")
    st.scatter_chart(filtered_df[['payload_mass', 'success']])

# ========== TAB 2 ==========
with tab2:
    st.subheader("🗺️ Launch Sites Map")
    launch_map = folium.Map(location=[20, 0], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(launch_map)

    for _, row in filtered_df.iterrows():
        outcome = "✅ Success" if row['success'] == 1 else "❌ Failed"
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"🚀 {row['name']}<br>📅 {row['date_utc']}<br>{outcome}",
            icon=folium.Icon(color="green" if row['success'] else "red", icon="rocket", prefix="fa")
        ).add_to(marker_cluster)

    st_data = st_folium(launch_map, width=900, height=500)

# ========== TAB 3 ==========
with tab3:
    st.subheader("🤖 Predict SpaceX Launch Success")

    st.markdown("Fill out launch details:")

    name = st.text_input("🧾 Mission Name", value="Demo Mission")
    date_utc = st.date_input("📅 Launch Date")

    rocket_encoded = st.selectbox("🚀 Rocket (Encoded)", options=[0, 1, 2], format_func=lambda x: f"Rocket {x}")
    launchpad_encoded = st.selectbox("🛰️ Launchpad (Encoded)", options=[0, 1, 2], format_func=lambda x: f"Launchpad {x}")
    payload_mass = st.number_input("📦 Payload Mass (kg)", min_value=0.0, max_value=50000.0, value=6000.0)

    lon = st.number_input("🌍 Longitude", min_value=-180.0, max_value=180.0, value=-80.5772)
    lat = st.number_input("🌎 Latitude", min_value=-90.0, max_value=90.0, value=28.5619)

    temperature = st.number_input("🌡️ Temperature (°C)", min_value=-100.0, max_value=100.0, value=25.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    wind_speed = st.number_input("💨 Wind Speed (m/s)", min_value=0.0, max_value=100.0, value=5.0)

    year = st.number_input("📅 Year", min_value=2002, max_value=2030, value=2020)
    month = st.number_input("📆 Month", min_value=1, max_value=12, value=6)
    day = st.number_input("📅 Day", min_value=1, max_value=31, value=15)
    hour = st.number_input("⏰ Hour", min_value=0, max_value=23, value=13)

    if st.button("🔍 Predict Launch Success"):
        input_data = np.array([[rocket_encoded, launchpad_encoded, payload_mass,
                                lon, lat, temperature, humidity, wind_speed,
                                year, month, day, hour]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"✅ Predicted: **Successful Launch** with {probability * 100:.2f}% confidence.")
        else:
            st.error(f"❌ Predicted: **Failed Launch** with {(1 - probability) * 100:.2f}% confidence.")

# Footer
st.markdown("---")
st.markdown("Made with 💻 during Saylani Hackathon Night by **Zafir Abdullah**")

