import streamlit as st
import numpy as np
import pickle  

# Load pre-trained model and scaler
@st.cache_resource
def load_model():
    with open("loan_approval_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)  
    return model

# Predict loan eligibility
def predict_eligibility(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    
    prediction = model.predict(input_data)[0]  # 0 or 1 for ineligible or eligible
    probability = model.predict_proba(input_data)[0][1]  # Probability for eligible
    return prediction, probability

# Streamlit app
st.set_page_config(page_title="AgriFlow Loan Eligibility", page_icon=":bank:", layout="centered")

# App title
st.title("AgriFlow: Loan Eligibility Prediction App :money_with_wings:")

# Description
st.markdown("""
    **Welcome to AgriFlow's Loan Eligibility Prediction App!**  
    This app helps you determine whether applicant's are eligible for a loan based on their personal information.
    Fill in the details below and click on "Predict Loan Eligibility" to find out!
    """)

# Sidebar for user inputs with better grouping
st.sidebar.header("Enter Your Details")

# Grouping inputs into sections
st.sidebar.subheader("Personal Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Marital Status", ["Married", "Not Married"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

st.sidebar.subheader("Income and Loan Details")
Applicant_Income = st.sidebar.number_input("Applicant Income (₦)", min_value=0, max_value=10000000, value=5000)
Coapplicant_Income = st.sidebar.number_input("Coapplicant Income (₦)", min_value=0, max_value=5000000, value=2000)
Loan_Amount = st.sidebar.number_input("Loan Amount (₦)", min_value=0, max_value=10000000, value=100000)
Term = st.sidebar.number_input("Loan Term (months)", min_value=12, max_value=360, value=120)

st.sidebar.subheader("Other Information")
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

# Encoding categorical variables
Gender = True if gender == "Male" else False
Married = True if married == "Married" else False
Education = True if education == "Graduate" else False
Self_Employed = True if self_employed == "Yes" else False
Credit_History = True if credit_history == "Yes" else False

# Encoding dependents
Dependents_0 = True if dependents == "0" else False
Dependents_1 = True if dependents == "1" else False
Dependents_2 = True if dependents == "2" else False
Dependents_3_plus = True if dependents == "3+" else False

# Encoding area
Area_Rural = True if area == "Rural" else False
Area_Semiurban = True if area == "Semiurban" else False
Area_Urban = True if area == "Urban" else False

# Combine all inputs into a single list with matching column names
input_data = [
    Applicant_Income,
    Coapplicant_Income,
    Loan_Amount,
    Term,
    Credit_History,
    Gender,
    Married,
    Dependents_0,
    Dependents_1,
    Dependents_2,
    Dependents_3_plus,
    Education,
    Self_Employed,
    Area_Rural,
    Area_Semiurban,
    Area_Urban,
]

# Load model
model = load_model()

# Predict loan eligibility
if st.button("Predict Loan Eligibility", use_container_width=True):
    prediction, probability = predict_eligibility(model, input_data)

    # Display results with improved visuals
    if prediction == 1:
        st.success(f"🎉 Congratulations! You are **eligible** for the loan with a probability of **{probability:.2%}**.")
    else:
        st.error(f"❌ Unfortunately, you are **not eligible** for the loan. Probability of eligibility: **{probability:.2%}**.")

# Footer
st.markdown("""
    ---
    **Disclaimer:** This tool is for prediction purposes only and should not be used as the sole basis for financial decisions.
    """)
