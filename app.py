import streamlit as st
from model_predict import predict_water_quality
from utilities import load_water_data, split_scale_data, split_data, scale_data, create_lag_features
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str




file_rawData = "Complete_Data_WQI.xlsx"

df_water_data = load_water_data(file_rawData)
print('=== Loaded raw water data successfully!!! ===')

features = ['Sulfate (mg/L)', 'Chloride (mg/L)', 'Sodium (mg/L)', 'Potassium (mg/L)',
                'Calcium (mg/L)', 'Magnesium (mg/L)', 'Total Dissolved Solids (mg/L)',
                'Turbidity (NTU)', 'Temperature (deg C)', 'pH',
                'Dissolved Oxygen (mg/L)', 'Nitrate (mg/L)', 'Fecal Coliform (cfu/100ml)']


# X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, dates_trian, dates_test = split_scale_data(df_water_data)
X_train, X_test, y_train, y_test, dates_trian, dates_test = split_data(df_water_data)
print('=== Splited data successfully!!! ===')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

selected_df_water_data = df_water_data.tail(4).copy()
selected_df_water_data = selected_df_water_data.drop(columns=['WQI', 'ActivityStartDate'])


# df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)
# print('Shape of df_X_test_scaled: ', df_X_test_scaled.shape)
df_y_test = y_test.to_frame('WQI')
print('Shape of df_y_test: ', df_y_test.shape)
df_dates_test = dates_test.to_frame('ActivityStartDate')
print('Shape of df_dates_test: ', df_dates_test.shape)



st.set_page_config(page_title="Water Quality Index (WQI) Predictor", layout="wide")

# # --- BANNER IMAGE AT THE TOP ---
# banner = Image.open("AI-and-Water.jpg")  # Replace with your banner file
# st.image(banner, use_column_width=True)


st.markdown("<h2 style='text-align: center; color: white;'>🌊  AI-Powered Dashboard For Water Quality Prediction  🌊</h2>", unsafe_allow_html=True)
# st.markdown(
#     """
#     <p style='text-align: center; font-size: 24px;'>
#         <span style='color: white;'>William Sun, Newark Academy</span>
#     </p>
#     """,
#     unsafe_allow_html=True
# )

st.markdown(
    """
    <style>
    /* Set overall app background */
    .stApp {
        background-color: #003b6b;
    }

    /* Hide top-right menu and Streamlit footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
# background-color: #00325B;



st.markdown("""
    <style>
    div.stButton > button:first-child {
        font-size: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        background-color: #249628; /* Green */
    }
    div.stButton > button:first-child > span {
        font-size: 18px; /* Adjust icon size here */
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
    .streamlit-expanderHeader > div > div > span {
        font-size: 30px !important; /* Or any desired size */
    }
    </style>
""", unsafe_allow_html=True)



with st.container():
    # st.title("Driven by XGBoost With Lag Features (Past 3 Days)")

    if X_test is not None:
        if len(X_test) < 3:
            st.error("Raw data must contain at least 3 rows to extract lag features.")
        else:
            col = st.columns((2, 4, 2), gap='medium')
            with col[0]:
                # st.markdown('#### ➤ Introduction')
                st.markdown('<h4 style="color:white;">➤ Introduction</h4>', unsafe_allow_html=True)

                # st.markdown("""
                #     - Data: [Delaware River Basin](https://www.sciencebase.gov/catalog/item/5e010424e4b0b207aa033d8c)
                #     - Predict Water Quality Index (WQI) using AI
                #     - Powered by XGBoost model
                #     - Uses actual 13 environmental features with lag enhancements
                #     - Real-time prediction
                #     """)
                st.markdown("""
                    <ul style="color:white; font-size:16px;">
                        <li>Data: <a href="https://www.usgs.gov/data/multi-source-surface-water-quality-data-and-us-geological-survey-streamgage-match-delaware" target="_blank">Delaware River Basin</a></li>
                        <li>Predict Water Quality Index (WQI) using AI</li>
                        <li>Powered by XGBoost model</li>
                        <li>Uses actual 13 environmental features with lag enhancements</li>
                        <li>Real-time prediction</li>
                    </ul>
                """, unsafe_allow_html=True)

                # Spacer
                # st.markdown("---")

                # with st.expander('About', expanded=True):
                #     st.write('''
                #         William Sun, 8th Grade, Newark Academy
                #         ''')

                st.markdown("""
                    <style>
                        div.streamlit-expanderHeader {
                            color: white;
                        }
                    </style>
                """, unsafe_allow_html=True)

                with st.expander('About', expanded=True):
                    st.markdown("""
                        <p style="color:white;">
                            William Sun, Newark Academy
                        </p>
                    """, unsafe_allow_html=True)

                    
            input_values = []
        
            with col[1]:
                # st.markdown("#### ➤ Input Parameters (with Lag Features)", unsafe_allow_html=True)
                st.markdown("""
                    <div style='font-size:22px;font-weight: bold;color: white;'>
                        ➤ Input Parameters <span style='font-size:16px;'>(with Lag Features)</span>
                    </div>
                    <div style='font-size:16px;color: white;'>
                        (Parameters can be adjusted by clicking '-' or '+')
                    </div>
                """, unsafe_allow_html=True)

                
                with st.container():
                    title_col1, title_col2 = st.columns([2, 2])
                    with title_col1:
                        for i in range(0, len(features)//2):
                            input_values.append(st.number_input(X_test.columns[i], value=X_test.iloc[-1, i], disabled=False) )
                    with title_col2:
                        for i in range(len(X_test.columns)//2, len(X_test.columns)):
                            input_values.append(st.number_input(X_test.columns[i], value=X_test.iloc[-1, i], disabled=False) )

                    print('input_values: ', input_values)

                    df_input_values = pd.DataFrame(np.array(input_values).reshape(1, -1), columns=features)
                    print('Shape of df_input_values: ', df_input_values.shape)
                    # print('df_input_values:', df_input_values)
                    arr_latest_features = scale_data(X_train, df_input_values)
                    print('arr_latest_features:', arr_latest_features)


            with col[2]:
                button_predict_wqi = st.button("#### 🔍 Predict WQI \n👉 Click Me!")
                output_placeholder = st.empty()

                if button_predict_wqi:                    
                    prediction = predict_water_quality(arr_latest_features,'xgb') 

                    if prediction < 25:
                        quality = "🌟 Excellent Water Quality 😊👍"
                        color = "#1bb320"
                    elif prediction < 50:
                        quality = "✅ Good Water Quality 🙂"
                        color = "#1bb320"
                    elif prediction < 75:
                        quality = "⚠️ Poor Water Quality 😟"
                        color = "#FFD700"  # Yellow
                    elif prediction < 100:
                        quality = "🚨 Very Poor Water Quality 😢"
                        color = "#FFD700"  # Yellow
                    else:
                        quality = "❌ Unsuitable For Drinking 🚱"
                        color = "red"   

                    st.markdown(f"""
                        <div style='text-align: left;'>
                            <span style='color: white; font-size: 22px;'>Predicted WQI: </span>
                            <span style='color: {color}; font-size: 30px; font-weight: bold;'> {prediction:.0f}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"<p style='color:{color};font-size:22px;font-weight:bold;'>{quality}</p>", unsafe_allow_html=True)

                else:
                    # Reserve the same vertical space with invisible content (transparent text)
                    with output_placeholder:
                        st.markdown(f"""
                            <div style='visibility: hidden;'>
                                <span style='font-size: 22px;'>Predicted WQI: 000</span><br>
                                <span style='font-size: 30px;'>Sample Quality</span>
                            </div>
                        """, unsafe_allow_html=True)
                # Spacer
                # st.markdown("---")

                image = Image.open("drb_illustration.jpg")
                st.markdown(
                    f"""
                    <img src='data:image/png;base64,{image_to_base64(image)}' style='width: 60%; max-width: 500px; height: auto;' />
                    """,
                    unsafe_allow_html=True
                )

                # Custom left-aligned caption
                st.markdown(
                    """
                    <p style='text-align: left; font-size: 15px;'>
                        <span style='color: grey;'>Delaware River Basin\nSource:</span>
                        <a href='https://www.nj.gov/drbc/basin/' target='_blank' style='color: #1a73e8; text-decoration: underline;'>NJ.gov</a>
                    </p>
                    """,
                    unsafe_allow_html=True
                )




            # st.subheader("📈 WQI Trend (Past 10 Days)")

            # fig, ax = plt.subplots()
            # ax.plot(df_dates_test['ActivityStartDate'], df_y_test['WQI'], color='blue', marker='o')
            # ax.set_xlabel("Date")
            # ax.set_ylabel("WQI")
            # ax.set_title("10-Day WQI Trend")
            # ax.grid(True)
            # st.pyplot(fig)

