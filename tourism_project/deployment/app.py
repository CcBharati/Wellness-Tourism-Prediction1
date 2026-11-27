import os
import joblib
import pandas as pd
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="🏖️",
    layout="wide"
)

# 1. Title and Description
st.title("🏖️ Wellness Tourism Package Purchase Predictor")
st.markdown("""
Predict whether a customer is likely to purchase the new **Wellness Tourism Package**
based on their profile and sales interaction details.
""")

# 2. Load the Trained Model Artifact
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please run the model training step first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# 3. Sidebar / Input Form Structure
st.sidebar.header("📋 Customer Input Parameters")

def get_user_inputs():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Customer Demographic Profile")
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=100000, value=25000, step=1000)
        city_tier = st.selectbox("City Tier", [1, 2, 3])

    with col2:
        st.subheader("✈️ Travel & Interaction Details")
        typeof_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
        num_person = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        num_trips = st.number_input("Average Annual Trips", min_value=1, max_value=20, value=3)
        passport = st.selectbox("Has Valid Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        num_children = st.number_input("Number of Children Below Age 5", min_value=0, max_value=5, value=0)
        pref_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5])

        st.subheader("📞 Sales Interaction Details")
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        duration_pitch = st.number_input("Duration of Pitch (mins)", min_value=1, max_value=120, value=15)
        num_followups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, value=3)

    # Compile input values into a single-row DataFrame
    input_data = {
        'Age': [age],
        'TypeofContact': [typeof_contact],
        'CityTier': [city_tier],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [num_person],
        'PreferredPropertyStar': [pref_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [num_trips],
        'Passport': [passport],
        'OwnCar': [own_car],
        'NumberOfChildrenVisiting': [num_children],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income],
        'PitchSatisfactionScore': [pitch_score],
        'ProductPitched': [product_pitched],
        'NumberOfFollowups': [num_followups],
        'DurationOfPitch': [duration_pitch]
    }
    return pd.DataFrame(input_data)

user_df = get_user_inputs()

# Display input summary
with st.expander("🔍 View Input Summary Dataframe", expanded=False):
    st.dataframe(user_df)

st.divider()

# 4. Make Prediction
if st.button("🚀 Predict Purchase Likelihood", type="primary"):
    try:
        prediction = model.predict(user_df)[0]
        probability = model.predict_proba(user_df)[0][1]

        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.success(f"**High Potential Customer!** The customer is **likely to purchase** the Wellness Package. (Confidence: {probability:.1%})")
            st.balloons()
        else:
            st.warning(f"**Low Potential Customer.** The customer is **unlikely to purchase** the package. (Confidence: {(1 - probability):.1%})")

        st.progress(float(probability), text=f"Purchase Probability: {probability:.1%}")

    except Exception as e:
        st.error(f"Error executing prediction: {e}")
