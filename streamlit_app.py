import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download, HfApi

# --- Hugging Face Model Hub Configuration ---
MODEL_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "nareshpaib/tourism_classifier")
MODEL_FILE_NAME = "random_forest_model.joblib"

# --- Function to load the model ---
@st.cache_resource
def load_model():
    try:
        # Download model from Hugging Face Hub
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILE_NAME)
        model = joblib.load(model_path)
        st.success(f"Model '{MODEL_FILE_NAME}' loaded successfully from {MODEL_REPO_ID}.")
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        return None

model = load_model()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")
st.title("🌍 Tourism Package Purchase Predictor")
st.markdown("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

if model is None:
    st.warning("Model could not be loaded. Please ensure the model exists on Hugging Face Hub and your HF_TOKEN is correctly set in the environment or Colab secrets.")
else:
    # --- User Input Fields ---
    st.header("Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        typeofcontact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Government Service', 'Unemployed'])
        gender = st.selectbox("Gender", ['Male', 'Female'])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)

    with col2:
        productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium'])
        preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
        maritalstatus = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0, max_value=50, value=5)
        passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        owncar = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
        designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
        monthlyincome = st.number_input("Monthly Income (USD)", min_value=0.0, value=5000.0)
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, value=10.0)

    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [typeofcontact],
        'CityTier': [citytier],
        'DurationOfPitch': [durationofpitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [numberofpersonvisiting],
        'NumberOfFollowups': [numberoffollowups],
        'ProductPitched': [productpitched],
        'PreferredPropertyStar': [preferredpropertystar],
        'MaritalStatus': [maritalstatus],
        'NumberOfTrips': [numberoftrips],
        'Passport': [passport],
        'PitchSatisfactionScore': [0],
        'OwnCar': [owncar],
        'NumberOfChildrenVisiting': [numberofchildrenvisiting],
        'Designation': [designation],
        'MonthlyIncome': [monthlyincome]
    })

    # Define all possible categorical values based on training data to ensure consistent one-hot encoding
    all_typeofcontact = ['Self Inquiry', 'Company Invited']
    all_occupation = ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Government Service', 'Unemployed']
    all_gender = ['Male', 'Female']
    all_productpitched = ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium']
    all_maritalstatus = ['Married', 'Single', 'Divorced']
    all_designation = ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director']

    # Apply one-hot encoding to input data
    input_data_encoded = pd.get_dummies(input_data, columns=['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation'])

    # Align columns with the training data columns
    training_columns = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
        'PreferredPropertyStar', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
        'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
        'TypeofContact_Self Inquiry', 'Occupation_Freelancer', 'Occupation_Government Service', 'Occupation_Large Business',
        'Occupation_Salaried', 'Occupation_Small Business', 'ProductPitched_Deluxe', 'ProductPitched_King',
        'ProductPitched_Premium', 'ProductPitched_Standard', 'ProductPitched_Super Deluxe',
        'MaritalStatus_Married', 'MaritalStatus_Single', 'Designation_Executive', 'Designation_Manager',
        'Designation_Senior Manager', 'Designation_VP', 'Gender_Male'
    ]

    # Add missing columns with 0
    for col in training_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    
    # Ensure the order of columns is the same as in training data
    input_data_processed = input_data_encoded[training_columns]

    # --- Prediction Button ---
    if st.button("Predict Purchase"):
        if model:
            try:
                prediction = model.predict(input_data_processed)[0]
                prediction_proba = model.predict_proba(input_data_processed)[0]

                st.subheader("Prediction Result")
                if prediction == 1:
                    st.success(f"The customer is likely to purchase the package! (Probability: {prediction_proba[1]*100:.2f}%) 🥳")
                else:
                    st.info(f"The customer is not likely to purchase the package. (Probability: {prediction_proba[0]*100:.2f}%) 😔")

                st.write("**Prediction Details:**")
                st.write(f"Purchase Probability (No): {prediction_proba[0]*100:.2f}%")
                st.write(f"Purchase Probability (Yes): {prediction_proba[1]*100:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.error("Cannot make prediction as the model was not loaded successfully.")
