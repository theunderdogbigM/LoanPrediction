import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.header("Loan Prediction")
st.title("Loan Prediction")

data = pd.read_csv('loan.csv')


data['dependences'] = pd.to_numeric(data['dependences'], errors='coerce')


data.dropna(subset=['dependences'], inplace=True)

X = data[['gender', 'marital_status', 'dependences', 'education']]
y = data.iloc[:, [4]]

label_encoder = LabelEncoder()

X['gender'] = label_encoder.fit_transform(X['gender'])
X['marital_status'] = label_encoder.fit_transform(X['marital_status'])
X['education'] = label_encoder.fit_transform(X['education'])
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


gender = st.number_input("Please specify gender", min_value=0, max_value=1, step=1, value=0)
married = st.number_input("Please specify maritial status", min_value=0, max_value=1, step=1, value=0)
dependences = st.number_input("Please specify dependences", min_value=0, max_value=10, step=1, value=0)
education = st.number_input("Please specify education", min_value=0, max_value=1, step=1, value=0)
loan_status = st.number_input("Please specify loan_status", min_value=0, max_value=1, step=1, value=0)

if st.button("Submit"):
    try:
        gender_encoded = label_encoder.transform([gender])[0]
        marital_encoded = label_encoder.transform([married])[0]
        education_encoded = label_encoder.transform([education])[0]
        loan_encoded = label_encoder.transform([loan_status])[0]

        features = [['gender', 'marital_status', 'dependences', 'education']]


        prediction = model.predict(features)

        if prediction == 1:
            result = "Congratulations, you're eligible for the loan!"
        else:
            result = "We're sorry, your loan request has not been approved."
        st.write("Loan Repayment Prediction:", result)

    except ValueError as e:
        st.write("An error occurred:", e)
