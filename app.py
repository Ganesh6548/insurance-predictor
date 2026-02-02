
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("insurance_model.pkl")
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_model()
    features = metadata["features"]
    target = metadata["target"]
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Predict insurance charges based on your profile")

# Sidebar inputs
st.sidebar.header("Your Profile")

# Collect inputs
inputs = {}

# Demographic
inputs["age"] = st.sidebar.slider("Age", 18, 65, 30)
inputs["bmi"] = st.sidebar.slider("BMI", 15.0, 40.0, 25.0, 0.1)
inputs["children"] = st.sidebar.slider("Children", 0, 5, 0)

# Lifestyle
if "sex_male" in features:
    sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
    inputs["sex_male"] = 1 if sex == "Male" else 0

if "smoker_yes" in features:
    smoker = st.sidebar.selectbox("Smoker", ["No", "Yes"])
    inputs["smoker_yes"] = 1 if smoker == "Yes" else 0

# Medical
if "claim_amount" in features:
    inputs["claim_amount"] = st.sidebar.slider("Claim Amount ($)", 0, 50000, 10000, 100)

if "annual_salary" in features:
    inputs["annual_salary"] = st.sidebar.slider("Annual Salary ($)", 20000, 150000, 50000, 1000)

# Add other features with defaults
for feature in features:
    if feature not in inputs and not feature.startswith(("sex_", "smoker_", "region_")):
        inputs[feature] = 0

# Prepare input for prediction
input_df = pd.DataFrame([inputs])

# Add missing features
for feature in features:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder columns
input_df = input_df[features]

# Make prediction
if st.sidebar.button("Predict Cost", type="primary"):
    prediction = model.predict(input_df)[0]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Charges", f"${prediction:,.2f}")
    
    with col2:
        st.metric("Model R² Score", f"{metadata['test_r2']:.3f}")
    
    with col3:
        st.metric("Features Used", len(features))
    
    # Feature importance
    st.subheader("Feature Impact")
    
    # Calculate contributions
    contributions = []
    for i, feature in enumerate(features):
        coef = model.coef_[i]
        value = input_df[feature].iloc[0]
        impact = coef * value
        contributions.append({
            "feature": feature,
            "impact": abs(impact)
        })
    
    # Sort and display top 5
    contributions.sort(key=lambda x: x["impact"], reverse=True)
    top_5 = contributions[:5]
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    feature_names = [c["feature"].replace("_", " ").title() for c in top_5]
    impacts = [c["impact"] for c in top_5]
    
    bars = ax.barh(feature_names, impacts)
    ax.set_xlabel("Impact on Prediction ($)")
    ax.set_title("Top 5 Influencing Factors")
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + max(impacts)*0.01, bar.get_y() + bar.get_height()/2,
                f"${width:,.0f}", ha="left", va="center")
    
    st.pyplot(fig)
    
    # Input summary
    st.subheader("Your Input Summary")
    
    summary_data = []
    for key, value in inputs.items():
        if value != 0 or key in ["age", "bmi", "children"]:
            display_name = key.replace("_", " ").title()
            if isinstance(value, (int, np.integer)):
                display_value = str(value)
            elif isinstance(value, float):
                if key in ["annual_salary", "claim_amount"]:
                    display_value = f"${value:,.0f}"
                else:
                    display_value = f"{value:.1f}"
            else:
                display_value = str(value)
            summary_data.append([display_name, display_value])
    
    summary_df = pd.DataFrame(summary_data, columns=["Parameter", "Value"])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if "smoker_yes" in inputs and inputs["smoker_yes"] == 1:
        st.info("Quitting smoking could significantly reduce your insurance costs.")
    
    if inputs.get("bmi", 25) > 25:
        st.info("Maintaining a healthy BMI could lower your insurance premiums.")
    
    if inputs.get("children", 0) > 2:
        st.info("Consider family insurance plans for potential savings.")

# Footer
st.markdown("---")
st.caption("Note: This is a predictive model for demonstration purposes.")
