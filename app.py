import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load artifacts
model = pickle.load(open("model.pkl", "rb"))
ohe = pickle.load(open("onehot_encoder.pkl", "rb"))
num_imp = pickle.load(open("num_imputer.pkl", "rb"))
cat_imp = pickle.load(open("cat_imputer.pkl", "rb"))
le_edu = pickle.load(open("le_education.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
category_values = pickle.load(open("category_values.pkl", "rb"))


# UI
st.title("CrediSure – Loan Approval System")
st.write("Enter applicant details to check loan approval")


# User Inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0)
Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
Age = st.number_input("Age", min_value=18, max_value=100)
Dependents = st.number_input("Dependents", min_value=0)
Credit_Score = st.number_input("Credit Score", min_value=0)
Existing_Loans = st.number_input("Existing Loans", min_value=0)
DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0)
Savings = st.number_input("Savings", min_value=0)
Collateral_Value = st.number_input("Collateral Value", min_value=0)
Loan_Amount = st.number_input("Loan Amount", min_value=0)
Loan_Term = st.number_input("Loan Term (months)", min_value=0)

Employment_Status = st.selectbox(
    "Employment Status",
    category_values["Employment_Status"]
)
Marital_Status = st.selectbox(
    "Marital Status",
    category_values["Marital_Status"]
)
Loan_Purpose = st.selectbox(
    "Loan Purpose",
    category_values["Loan_Purpose"]
)
Property_Area = st.selectbox(
    "Property Area",
    category_values["Property_Area"]
)
Gender = st.selectbox(
    "Gender",
    category_values["Gender"]
)
Employer_Category = st.selectbox(
    "Employer Category",
    category_values["Employer_Category"]
)
Education_Level = st.selectbox(
    "Education Level",
    le_edu.classes_.tolist()
)

# Build input DataFrame
input_df = pd.DataFrame([{
    "Applicant_Income": Applicant_Income,
    "Coapplicant_Income": Coapplicant_Income,
    "Employment_Status": Employment_Status,
    "Age": Age,
    "Marital_Status": Marital_Status,
    "Dependents": Dependents,
    "Credit_Score": Credit_Score,
    "Existing_Loans": Existing_Loans,
    "DTI_Ratio": DTI_Ratio,
    "Savings": Savings,
    "Collateral_Value": Collateral_Value,
    "Loan_Amount": Loan_Amount,
    "Loan_Term": Loan_Term,
    "Loan_Purpose": Loan_Purpose,
    "Property_Area": Property_Area,
    "Education_Level": Education_Level,
    "Gender": Gender,
    "Employer_Category": Employer_Category
}])


# Preprocessing (NO SCALING)
input_df[num_imp.feature_names_in_] = num_imp.transform(
    input_df[num_imp.feature_names_in_]
)
input_df[cat_imp.feature_names_in_] = cat_imp.transform(
    input_df[cat_imp.feature_names_in_]
)

input_df["Education_Level"] = le_edu.transform(
    input_df["Education_Level"]
)

encoded = ohe.transform(input_df[ohe.feature_names_in_])
encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(ohe.feature_names_in_),
    index=input_df.index
)

input_df = pd.concat(
    [input_df.drop(columns=ohe.feature_names_in_), encoded_df],
    axis=1
)

input_df = input_df.reindex(columns=features, fill_value=0)


# Prediction (GaussianNB)
if st.button("Check Loan Approval"):
    proba = model.predict_proba(input_df)[0][1]
    st.info(f"Approval Probability: {proba:.2%}")

    if proba >= 0.4:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
