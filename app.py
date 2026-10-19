
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Configure webpage
st.set_page_config(
    page_title="Investment Profit Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Investment Profit Predictor")
st.write(
    "Enter your investment details to estimate "
    "the expected profit using Machine Learning."
)

# Load trained model
model_path = Path(__file__).parent / "model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("model.pkl not found. Run train_model.py first.")
    st.stop()

# Investment inputs
st.subheader("Investment Details")

rd_spend = st.number_input(
    "R&D Spend", min_value=0.0,
    value=100000.0, step=1000.0
)

administration = st.number_input(
    "Administration Spend", min_value=0.0,
    value=100000.0, step=1000.0
)

marketing = st.number_input(
    "Marketing Spend", min_value=0.0,
    value=100000.0, step=1000.0
)

state = st.selectbox(
    "Select State",
    ["California", "Florida", "New York"]
)

# Predict profit
if st.button("Predict Profit", type="primary"):

    # Start with the original input fields
    input_data = pd.DataFrame([{
        "R&D Spend": rd_spend,
        "Administration": administration,
        "Marketing Spend": marketing,
        "State": state
    }])

    # Apply the column transformations used in training
    input_data = input_data.rename(columns={
        "R&D Spend": "Research",
        "Administration": "Promotion",
        "Marketing Spend": "DigitalMarketing"
    })

    # One-hot encode the State column
    input_data = pd.get_dummies(
        input_data,
        columns=["State"],
        dtype=int
    )
    
    
    
    # Match the exact columns and order used to train the model
if hasattr(model, "feature_names_in_"):
    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )


 

    try:
        prediction = model.predict(input_data)[0]

        st.success(f"Estimated Profit: {prediction:,.2f}")

        st.caption(
            "This is a model estimate, not a guarantee "
            "of actual profit."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
