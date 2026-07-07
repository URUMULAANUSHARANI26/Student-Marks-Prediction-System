import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(page_title="Student AI App", page_icon="🎓", layout="centered")
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["Predict", "About"])

if page == "About":
    st.write("This project predicts student marks using Machine Learning.")
model = pickle.load(open("model/model.pkl", "rb"))
data = pd.read_csv("Data/student_data.csv")

st.title("🎓 Student Performance Predictor")

hours = st.slider("Study Hours", 0, 10)
attendance = st.slider("Attendance %", 0, 100)

if st.button("Predict"):
    result = model.predict([[hours, attendance]])
    st.success(f"Predicted Marks: {result[0]:.2f}")

# 📊 Graph
st.subheader("📊 Data Visualization")
plt.scatter(data["Hours_Studied"], data["Marks"])
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
st.pyplot(plt)