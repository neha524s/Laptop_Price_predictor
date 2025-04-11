import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.markdown("""
    <style>
        .main {
            background-color: #f0f8ff; 
            color: #333333;
        }

        h1 {
            color: #1f77b4; 
        }

        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

pipe = pickle.load(open('pipe1.pkl', 'rb'))
df = pickle.load(open('df1.pkl', 'rb'))

st.title("Laptop Price Predictor")

company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())  # Renamed from 'type'
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size', value=15.6)
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                           '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu-name'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu-name'].unique())
os = st.selectbox('OS', df['Op-sys'].unique())

if st.button('Predict Price'):
    # Convert categorical variables to numeric
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Clipping PPI values to match training data
    ppi = max(82.313, min(ppi, 202.373))

    # Encode categorical features as numbers using index mapping
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [laptop_type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'IPS': [ips],
        'PPI': [ppi],
        'Cpu-name': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu-name': [gpu],
        'Op-sys': [os]
    })

    # Debugging print
    print("Input DataFrame:\n", input_data)

    try:
        predicted_price = np.exp(pipe.predict(input_data))
        st.title(f"Predicted Price: {int(predicted_price)} INR")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
