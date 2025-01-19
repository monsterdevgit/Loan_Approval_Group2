# Loan Approval Prediction Model

This repository contains the code and resources for predicting loan approval eligibility using machine learning.

## Files in this Repository

- **Model.ipynb**: Jupyter notebook that contains the model training process.
- **loan_approval_model.pkl**: Pre-trained model for loan approval prediction.
- **loan_test.csv**: Test dataset used to evaluate the model.
- **loan_train.csv**: Training dataset used to build the model.
- **requirements.txt**: Python dependencies required for the project.
- **streamlit_app.py**: Streamlit application for deploying the loan approval prediction model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/monsterdevgit/Loan_Approval_Group2.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```

## How It Works

This project uses various factors such as applicant income, coapplicant income, credit history, loan amount, and demographic details to predict loan approval. The machine learning model was trained using the provided datasets and is used to make predictions on user inputs via a Streamlit interface.

## Usage

Once the Streamlit app is running, enter the required information through the sidebar and click "Predict" to see if the loan is approved along with the probability.

