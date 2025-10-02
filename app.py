import streamlit as st      # Streamlit for the web app
import pandas as pd         # Pandas for data handling
import joblib               # Joblib for loading/saving models

# Load trained pipeline
model = joblib.load("fraud_detection_pipeline.pkl")

st.title("💳 Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and click **Predict**")

st.divider()

# ---- User Inputs ----
transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=1000.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=1000.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=1000.0)

# ---- Prediction ----
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction : '{int(prediction)}'")

    if prediction == 1:
        st.error("🚨 This transaction can be FRAUD!")
    else:
        st.success("✅ This transaction looks like it is not a fraud")

   