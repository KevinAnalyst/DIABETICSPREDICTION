import streamlit as st
import numpy as np
import joblib
#Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.title("DIABETIES DETECTION AND PREDICTION USING MACHINE LEARNING & AI")
st.subheader("This webapp is tailored in detecting diabetes by providing factors required")

Pregnancies=st.number_input("Enter the pregnancies test")

Glocose=st.number_input("Enter the Gluose level")

BloodPressure=st.number_input("Enter the Blood Pressure value")

SkinThicken=st.number_input("Enter the skin thickness value")

Insulin=st.number_input("Enter the insulin level")

BMI=st.number_input("Enter the BMI value")

DiabetesPedigreenFunction=st.number_input("Enter the daibetespedigreenfunction")

Age=st.number_input("Enter age")

diabetic=[Pregnancies,Glocose,BloodPressure,SkinThicken,Insulin,BMI,DiabetesPedigreenFunction,Age]
diabetic=np.asarray(diabetic)
diabetic=diabetic.reshape(-1,1)

def predict(diabetic):
    diabetic=joblib.load('model_logistics.sav')
    return diabetic.predict(diabetic)

if st.button("Diabetic prediction button"):
    st.success("Result outcome")
    st.text(diabetic[0])
    print("Diabetic")

