import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="Customer Satisfaction Predictor", layout="centered")

# load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    model = load_model()
except:
    st.error("Save the model first.")

# Heading
st.title("📊 Customer Satisfaction Prediction")
st.write(f"**Model Accuracy:** `0.2238` (Best Baseline Model)")
st.markdown("---")

# information from users
st.subheader("Enter Ticket Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    
with col2:
    priority = st.selectbox("Ticket Priority", ["Low", "Medium", "High", "Critical"])

ticket_type = st.selectbox("Ticket Type", ["Technical issue", "Billing inquiry", "Product inquiry", "Cancellation request"])

resolution_time = st.slider("Estimated Resolution Time (Hours)", 1, 168, 24)

# Prediction button
if st.button("Predict Satisfaction Rating"):
    # 1. Prepare the input in the EXACT same order as training
    # Example: if your model expects [Age, Resolution_Hrs]
    input_features = [[age, resolution_time]] 
    
    # 2. Convert to DataFrame (this helps keep column names consistent)
    input_df = pd.DataFrame(input_features, columns=['Customer Age', 'Resolution_Duration_Hrs'])
    
    # 3. Predict
    prediction = model.predict(input_df)
    
    # Display result
    rating = int(prediction[0])
    st.write(f"### Predicted Rating: {rating} / 5")
    st.write("⭐" * rating)

    # Messaging
    if rating >= 4:
        st.success("Beneficiary may be satisfied.")
    elif rating == 3:
        st.info("Average Satisfaction (Neutral Satisfaction).")
    else:
        st.warning("Beneficiary is depressed, give more attention.")

# Data statistics on side bar
st.sidebar.header("About the Project")
st.sidebar.info("Use RandomForestClassifier for this Model. Predict rating on the basis of customer tickets.")