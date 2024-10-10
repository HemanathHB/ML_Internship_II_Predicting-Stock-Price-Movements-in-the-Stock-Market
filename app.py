import streamlit as st
import pandas as pd
import joblib

st.title("Stock Price Movement Predictor")

# Load the model
model = joblib.load('stock_price_model.pkl')

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    if st.button('Predict'):
        # Prepare the input for prediction
        # Make sure to preprocess the input similar to training
        # For example, handle categorical variables and scaling if necessary

        # Predict
        predictions = model.predict(df)
        st.success(f"Predicted Price Movement: {predictions[-1]}")  # Display last prediction
