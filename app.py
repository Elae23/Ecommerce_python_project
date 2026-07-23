import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("customer_satisfaction_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Satisfaction Prediction System",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 Customer Satisfaction Prediction System")

st.markdown("""
### Machine Learning Powered E-Commerce Analytics

Predict whether a customer is likely to be satisfied based on their order details using a Random Forest machine learning model.
""")

st.divider()

# Sidebar
st.sidebar.title("Project Information")

st.sidebar.info("""
**Project:** Customer Satisfaction Prediction

**Model:** Random Forest Classifier

**Dataset:** Brazilian Olist E-commerce Dataset

**Developer:** Ashe'esla Arigu
""")

st.write(
    "Enter the customer's order details below to predict whether the customer is satisfied."
)

# Input columns
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("💰 Product Price", min_value=0.0)
    freight_value = st.number_input("🚚 Freight Value", min_value=0.0)
    payment_value = st.number_input("💳 Payment Value", min_value=0.0)
    payment_installments = st.number_input("💳 Payment Installments", min_value=1)

with col2:
    product_weight_g = st.number_input("📦 Product Weight (g)", min_value=0.0)
    product_length_cm = st.number_input("📏 Product Length (cm)", min_value=0.0)
    product_height_cm = st.number_input("📐 Product Height (cm)", min_value=0.0)
    product_width_cm = st.number_input("📐 Product Width (cm)", min_value=0.0)

# Prediction button
if st.button("🔍 Predict Customer Satisfaction", use_container_width=True):

    input_data = pd.DataFrame({
        "price": [price],
        "freight_value": [freight_value],
        "payment_value": [payment_value],
        "payment_installments": [payment_installments],
        "product_weight_g": [product_weight_g],
        "product_length_cm": [product_length_cm],
        "product_height_cm": [product_height_cm],
        "product_width_cm": [product_width_cm]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    confidence = probability[1] * 100

    st.divider()

    st.header("📊 Prediction Results")

    if prediction == 1:
        st.success("😊 Customer is likely to be SATISFIED")
    else:
        st.error("☹️ Customer is likely to be UNSATISFIED")

    st.metric("Confidence", f"{confidence:.2f}%")

    st.progress(float(probability[1]))

    st.subheader("📋 Customer Order Summary")
    st.dataframe(input_data)

# About section
with st.expander("ℹ️ About this Prediction Model"):
    st.write("""
This application predicts customer satisfaction using a Random Forest machine learning model trained on the Brazilian Olist E-commerce Dataset.

Features used:

- Product Price
- Freight Value
- Payment Value
- Payment Installments
- Product Weight
- Product Length
- Product Height
- Product Width
""")

st.divider()

st.markdown(
    """
    <center>
    <b>Developed by Ashe'esla Arigu</b><br>
    Customer Satisfaction Prediction System<br>
    Powered by Streamlit & Scikit-learn
    </center>
    """,
    unsafe_allow_html=True
)