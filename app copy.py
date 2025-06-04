import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from predict import predict_wqi

st.set_page_config(page_title="Water Quality Index Predictor", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-image: url("https://images.unsplash.com/photo-1502741126161-b048400d8a0d");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.title("🌊 Water Quality Index Predictor 🌊")

    uploaded_file = st.file_uploader("Upload a CSV file (with 13 features + WQI)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=['ActivityStartDate'])

        if len(df) < 3:
            st.error("CSV must contain at least 3 rows to extract lag features.")
        else:
            st.subheader("Input Features (Last 3 Days)")

            latest_features = df.iloc[-3:, 1:-1].values.flatten().reshape(1, -1)

            col1, col2 = st.columns(2)
            with col1:
                for i in range(0, len(df.columns[1:-1])//2):
                    st.text_input(df.columns[1+i], value=f"{df.iloc[-1, 1+i]:.2f}", disabled=True)
            with col2:
                for i in range(len(df.columns[1:-1])//2, len(df.columns[1:-1])):
                    st.text_input(df.columns[1+i], value=f"{df.iloc[-1, 1+i]:.2f}", disabled=True)

            if st.button("🔍 Predict WQI"):
                prediction = predict_wqi(latest_features)

                if prediction < 25:
                    quality = "🌟 Excellent Water Quality 😊👍"
                elif prediction < 50:
                    quality = "✅ Good Water Quality 🙂"
                elif prediction < 75:
                    quality = "⚠️ Poor Water Quality 😟"
                elif prediction < 100:
                    quality = "🚨 Very Poor Water Quality 😢"
                else:
                    quality = "❌ Unsuitable For Drinking 🚱"

                st.markdown(f"### Predicted WQI: `{prediction:.2f}`")
                st.markdown(f"## {quality}")

            st.subheader("📈 WQI Trend (Past 10 Days)")

            fig, ax = plt.subplots()
            ax.plot(df['ActivityStartDate'], df['WQI'], color='blue', marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("WQI")
            ax.set_title("10-Day WQI Trend")
            ax.grid(True)
            st.pyplot(fig)
