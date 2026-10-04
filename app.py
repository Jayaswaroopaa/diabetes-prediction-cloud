import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Load the trained SVM model
# --------------------------------------------------

with open("classifier.pkl", "rb") as file:
    classifier = pickle.load(file)


# --------------------------------------------------
# Load the original dataset
# Used to recreate the StandardScaler
# --------------------------------------------------

diabetes_dataset = pd.read_csv("diabetes.csv")

X = diabetes_dataset.drop(columns="Outcome")

scaler = StandardScaler()
scaler.fit(X)


# --------------------------------------------------
# Streamlit application
# --------------------------------------------------

st.title("Diabetes Prediction Using SVM")

st.write(
    "Enter the following information to predict whether "
    "the person is diabetic or not."
)


# --------------------------------------------------
# Input fields
# --------------------------------------------------

Pregnancies = st.number_input(
    "Number of Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

Glucose = st.number_input(
    "Glucose Level",
    min_value=0,
    max_value=300,
    value=120
)

BloodPressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

SkinThickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

Insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

BMI = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

Age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict"):

    input_data = (
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    )

    # Convert input into NumPy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for one prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize input using the same process as training
    std_data = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = classifier.predict(std_data)

    # Display result
    if prediction[0] == 0:
        st.success("The person is not diabetic.")
    else:
        st.error("The person is diabetic.")
