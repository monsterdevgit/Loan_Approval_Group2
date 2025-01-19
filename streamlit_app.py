import streamlit as st
import numpy as np
import pickle  

# Load pre-trained model and scaler
@st.cache_resource
def load_model():
    with open("loan_eligibility_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)  
    return model

# Predict loan eligibility
def predict_eligibility(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]  # 0 or 1 for ineligible or eligible
    probability = model.predict_proba(input_data)[0][1]  # Probability for eligible
    return prediction, probability

# Streamlit app
st.title("Loan Eligibility Prediction App")
st.write("This application predicts loan eligibility based on user inputs.")

# Sidebar for user inputs
st.sidebar.header("Input Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income (₦)", min_value=0, max_value=10000000, value=5000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income (₦)", min_value=0, max_value=5000000, value=2000)
loan_amount = st.sidebar.number_input("Loan Amount (₦)", min_value=0, max_value=10000000, value=100000)
term = st.sidebar.number_input("Loan Term (months)", min_value=12, max_value=360, value=120)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encoding categorical variables (adjust encoding based on your trained model)
gender_encoded = 1 if gender == "Male" else 0
married_encoded = 1 if married == "Yes" else 0
education_encoded = 1 if education == "Graduate" else 0
self_employed_encoded = 1 if self_employed == "Yes" else 0
credit_history_encoded = 1 if credit_history == "Yes" else 0

# Encoding dependents
dependents_0 = 1 if dependents == "0" else 0
dependents_1 = 1 if dependents == "1" else 0
dependents_2 = 1 if dependents == "2" else 0
dependents_3_plus = 1 if dependents == "3+" else 0

# Encoding area
area_rural = 1 if area == "Rural" else 0
area_semiurban = 1 if area == "Semiurban" else 0
area_urban = 1 if area == "Urban" else 0

# Combine all inputs into a single list
input_data = [
    applicant_income,
    coapplicant_income,
    loan_amount,
    term,
    credit_history_encoded,
    gender_encoded,
    married_encoded,
    dependents_0,
    dependents_1,
    dependents_2,
    dependents_3_plus,
    education_encoded,
    self_employed_encoded,
    area_rural,
    area_semiurban,
    area_urban,
]

# Load model
model = load_model()

# Predict loan eligibility
if st.button("Predict Loan Eligibility"):
    prediction, probability = predict_eligibility(model, input_data)

    # Display results
    if prediction == 1:
        st.success(f"Congratulations! You are eligible for the loan with a probability of {probability:.2%}.")
    else:
        st.error(f"Unfortunately, you are not eligible for the loan. Probability of eligibility: {probability:.2%}.")
