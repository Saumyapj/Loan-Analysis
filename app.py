
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_columns = joblib.load("model_columns.pkl")
model = joblib.load("loan_rf_model.pkl")
ohe = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")
scaling_cols = joblib.load("scaling_cols.pkl")


st.title("Loan Approval Prediction App")

Gender = st.selectbox("Gender",["Male", "Female"])
Married = st.selectbox("Married",["Yes", "No"])
Dependents = st.selectbox("Dependents",[0,1,2,3])
Education = st.selectbox("Education",["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed",["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income",min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income",min_value=0)
Loan_Amount = st.number_input("Loan Amount",min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term",min_value=0)
Credit_History = st.selectbox("Credit History",[0.0,1.0,2.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    
    input_df = pd.DataFrame({
        "Gender":[Gender],
        "Married":[Married],
        "Dependents":[Dependents],
        "Education":[Education],
        "Self_Employed":[Self_Employed],
        "ApplicantIncome":[ApplicantIncome],
        "CoapplicantIncome":[CoapplicantIncome],
        "LoanAmount":[Loan_Amount],
        "Loan_Amount_Term":[Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Property_Area":[Property_Area]})
    
    input_df['TotalIncome'] = input_df['ApplicantIncome'] + input_df['CoapplicantIncome']
    input_df["IncomeBand"] = pd.cut(input_df['TotalIncome'], bins=[ 1442,4166,5416.5,7521.75,81000], labels=False,include_lowest=True)
    input_df['LoanIncomeRatio'] = input_df['LoanAmount']/input_df['TotalIncome']
    input_df['HasCoapplicant'] = np.where(input_df['CoapplicantIncome'] > 0,1,0)
    input_df['ApplicantIncome'] = np.log1p(input_df['ApplicantIncome'])
    input_df['CoapplicantIncome'] = np.log1p(input_df['CoapplicantIncome'])
    input_df['LoanAmount'] = np.log1p(input_df['LoanAmount'])

    cat_cols=["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

    encoded_data = ohe.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_data,columns=ohe.get_feature_names_out(cat_cols), index=input_df.index)

    input_df = pd.concat([input_df, encoded_df], axis=1)
    input_df.drop(columns=cat_cols, inplace=True)

    input_df[scaling_cols] = scaler.transform(input_df[scaling_cols])
    
    final_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(final_df)

    if prediction == 'Y':
        st.success("Loan Status: Approved")
    else:
        st.error("Loan Status: Rejected")