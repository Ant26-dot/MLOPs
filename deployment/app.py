import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from huggingface_hub import hf_hub_download

# --- Page Configuration ---
st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    layout="centered"
)

# --- Constants ---
HF_MODEL_REPO_ID = "lomface/tourism-package-prediction-model"
MODEL_FILENAME = "model_0.joblib"
SCALER_FILENAME = "scaler.joblib"

# --- Load Model and Scaler from Hugging Face Model Hub ---
@st.cache_resource
def load_artifacts():
    try:
        # Download scaler
        scaler_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=SCALER_FILENAME, repo_type="model")
        scaler = joblib.load(scaler_path)

        # Download model
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type="model")
        model = joblib.load(model_path)

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model artifacts from Hugging Face: {e}")
        return None, None

model, scaler = load_artifacts()

# --- Title and Header ---
st.title("Wellness Tourism Package Purchase Predictor")
st.markdown("""
This internal application helps the marketing team identify customers who are most likely to purchase the newly introduced **Wellness Tourism Package** prior to outreach campaigns.
""")

if model is None or scaler is None:
    st.warning("Waiting for model and scaler artifacts to be successfully loaded from Hugging Face...")
else:
    st.sidebar.header("Pipeline Tuning")
    # Dynamic Classification Threshold adjustment to address class imbalance
    classification_threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Lowering this threshold increases Recall (captures more buyers but increases outreach costs). Raising it increases Precision (saves cost but may miss buyers)."
    )

    st.subheader("Enter Customer Details")

    # Organize inputs into columns with helpful descriptions
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (Customer age in years)", min_value=18, max_value=100, value=35)
        city_tier = st.selectbox("City Tier (1: High, 2: Mid, 3: Low standard of living)", options=[1, 2, 3], index=0)
        duration_pitch = st.number_input("Duration of Pitch (Minutes of sales presentation)", min_value=0.0, value=15.0, step=1.0)
        num_visits = st.number_input("Number of Persons Visiting (Total travelling party size)", min_value=1, max_value=10, value=2)
        num_followups = st.number_input("Number of Follow-ups (Sales agent callback attempts)", min_value=0, max_value=20, value=3)
        property_star = st.selectbox("Preferred Property Hotel Star Rating", options=[3, 4, 5], index=0)
        num_trips = st.number_input("Number of Annual Trips (Average vacations per year)", min_value=0, max_value=50, value=2)
        monthly_income = st.number_input("Monthly Income (Gross monthly income)", min_value=0.0, value=25000.0, step=500.0)

    with col2:
        passport = st.selectbox("Holds Valid Passport?", options=["No", "Yes"], index=0)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score (1: Poor, 5: Excellent)", min_value=1, max_value=5, value=3)
        own_car = st.selectbox("Owns Car?", options=["No", "Yes"], index=1)
        children_visiting = st.number_input("Number of Children Visiting (< Age 5)", min_value=0, max_value=5, value=1)

        typeof_contact = st.selectbox("Type of Contact", options=["Company Invited", "Self Inquiry"])
        occupation = st.selectbox("Occupation", options=["Salaried", "Small Business", "Large Business", "Freelancer"])
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Unmarried"])

    # Extra details mapped automatically behind the scenes to keep interface focused
    designation_mapped = "Executive"
    if age >= 50:
        designation_mapped = "VP"
    elif age >= 40:
        designation_mapped = "Senior Manager"
    elif age >= 30:
        designation_mapped = "Manager"

    # --- Construct Input DataFrame mirroring preprocessing features exactly ---
    input_data = {
        'Age': [float(age)],
        'CityTier': [float(city_tier)],
        'DurationOfPitch': [float(duration_pitch)],
        'NumberOfPersonVisiting': [float(num_visits)],
        'NumberOfFollowups': [float(num_followups)],
        'PreferredPropertyStar': [float(property_star)],
        'NumberOfTrips': [float(num_trips)],
        'Passport': [1.0 if passport == "Yes" else 0.0],
        'PitchSatisfactionScore': [float(pitch_satisfaction)],
        'OwnCar': [1.0 if own_car == "Yes" else 0.0],
        'NumberOfChildrenVisiting': [float(children_visiting)],
        'MonthlyIncome': [float(monthly_income)],
        'TypeofContact_Self Enquiry': [1.0 if typeof_contact == "Self Enquiry" else 0.0],
        'Occupation_Large Business': [1.0 if occupation == "Large Business" else 0.0],
        'Occupation_Salaried': [1.0 if occupation == "Salaried" else 0.0],
        'Occupation_Small Business': [1.0 if occupation == "Small Business" else 0.0],
        'Gender_Female': [1.0 if gender == "Female" else 0.0],
        'Gender_Male': [1.0 if gender == "Male" else 0.0],
        'ProductPitched_Deluxe': [0.0],
        'ProductPitched_King': [0.0],
        'ProductPitched_Standard': [0.0],
        'ProductPitched_Super Deluxe': [0.0],
        'MaritalStatus_Married': [1.0 if marital_status == "Married" else 0.0],
        'MaritalStatus_Single': [1.0 if marital_status == "Single" else 0.0],
        'MaritalStatus_Unmarried': [1.0 if marital_status == "Unmarried" else 0.0],
        'Designation_Executive': [1.0 if designation_mapped == "Executive" else 0.0],
        'Designation_Manager': [1.0 if designation_mapped == "Manager" else 0.0],
        'Designation_Senior Manager': [1.0 if designation_mapped == "Senior Manager" else 0.0],
        'Designation_VP': [1.0 if designation_mapped == "VP" else 0.0]
    }

    input_df = pd.DataFrame(input_data)

    # --- Scaling Preprocessing ---
    features_to_scale = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
        'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome'
    ]

    # Ensure safety check mapping only present features
    features_to_scale = [col for col in features_to_scale if col in input_df.columns]

    input_df_scaled = input_df.copy()
    input_df_scaled[features_to_scale] = scaler.transform(input_df[features_to_scale])

    # --- Make Predictions ---
    if st.button("Predict Purchase Likelihood", use_container_width=True):
        prediction_proba = model.predict_proba(input_df_scaled)[0][1]
        
        # Use our dynamic classification threshold 
        prediction = 1 if prediction_proba >= classification_threshold else 0

        st.markdown("--- ")
        st.subheader("Prediction Results")

        if prediction == 1:
            st.error(f"**Predicted Decision: Likely to Buy (ProdTaken = 1)**\n\nConfidence Score: **{prediction_proba * 100:.2f}%** (Threshold: {classification_threshold}) ")
            st.info("**Marketing Action:** Highly recommended for prioritized sales campaigns and personal outreach.")
        else:
            st.success(f"**Predicted Decision: Unlikely to Buy (ProdTaken = 0)**\n\nConfidence Score: **{(1 - prediction_proba) * 100:.2f}%** (Threshold: {classification_threshold})")
            st.info("**Marketing Action:** Keep in regular automated campaign pool; avoid high-cost direct sales outreach.")
