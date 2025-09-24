# dashboard.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="HR Prediction Dashboard", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: darkblue;'>💼 HR Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict Employee Attrition and Salary</p>", unsafe_allow_html=True)
st.markdown("---")

# Load HR dataset (first 30k rows)
df = pd.read_csv("HR_Data_MNC_Data Science Lovers.csv")
df = df.sample(n=30000, random_state=42).reset_index(drop=True)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df["Hire_Date"] = pd.to_datetime(df["Hire_Date"], errors="coerce")
df["Hire_Year"] = df["Hire_Date"].dt.year
df["Country"] = df["Location"].apply(lambda x: str(x).split(",")[-1].strip())

# Encode categorical columns
categorical_cols = ["Department","Job_Title","Work_Mode","Location","Country"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Target columns
df["Attrition"] = df["Status"].apply(lambda x: 1 if x in ["Resigned","Terminated"] else 0)

X_clf = df[["Department","Job_Title","Work_Mode","Location","Country","Experience_Years","Performance_Rating","Hire_Year"]]
y_clf = df["Attrition"]
X_reg = X_clf.copy()
y_reg = df["Salary_INR"]

# Scale features
scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)
scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

# Train models
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_clf_scaled, y_clf)
reg = RandomForestRegressor(n_estimators=200, random_state=42)
reg.fit(X_reg_scaled, y_reg)

# User Inputs
st.subheader("📝 Enter Employee Details")
col1, col2 = st.columns(2)

with col1:
    department = st.selectbox("Department", df["Department"].map(lambda x: label_encoders["Department"].inverse_transform([x])[0]).unique())
    job_title = st.selectbox("Job Title", df["Job_Title"].map(lambda x: label_encoders["Job_Title"].inverse_transform([x])[0]).unique())
    work_mode = st.selectbox("Work Mode", df["Work_Mode"].map(lambda x: label_encoders["Work_Mode"].inverse_transform([x])[0]).unique())
    location = st.selectbox("Location", df["Location"].map(lambda x: label_encoders["Location"].inverse_transform([x])[0]).unique())

with col2:
    country = st.selectbox("Country", df["Country"].map(lambda x: label_encoders["Country"].inverse_transform([x])[0]).unique())
    experience = st.number_input("Experience Years", min_value=0, max_value=50, value=5)
    performance = st.number_input("Performance Rating", min_value=1, max_value=5, value=3)
    hire_year = st.number_input("Hire Year", min_value=int(df["Hire_Year"].min()), max_value=int(df["Hire_Year"].max()), value=2020)

# Predict
if st.button("Predict"):
    input_data = {
        "Department": label_encoders["Department"].transform([department])[0],
        "Job_Title": label_encoders["Job_Title"].transform([job_title])[0],
        "Work_Mode": label_encoders["Work_Mode"].transform([work_mode])[0],
        "Location": label_encoders["Location"].transform([location])[0],
        "Country": label_encoders["Country"].transform([country])[0],
        "Experience_Years": experience,
        "Performance_Rating": performance,
        "Hire_Year": hire_year
    }
    input_df = pd.DataFrame([input_data])
    input_scaled_clf = scaler_clf.transform(input_df)
    input_scaled_reg = scaler_reg.transform(input_df)

    pred_attrition = clf.predict(input_scaled_clf)[0]
    pred_salary = reg.predict(input_scaled_reg)[0]

    st.markdown("---")
    if pred_attrition == 1:
        st.markdown(f"<h2 style='color: red;'>❌ Attrition Prediction: Will Leave</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='color: green;'>✅ Attrition Prediction: Will Stay</h2>", unsafe_allow_html=True)

    st.markdown(f"<h2 style='color: blue;'>💰 Predicted Salary: ₹{pred_salary:,.2f}</h2>", unsafe_allow_html=True)

# Mini Charts / Summary using Streamlit native charts
st.markdown("---")
st.subheader("📊 HR Summary Charts (from 30k dataset)")

col1, col2 = st.columns(2)

with col1:
    # Average salary per department
    dept_salary = df.groupby("Department")["Salary_INR"].mean().reset_index()
    dept_salary["Department"] = label_encoders["Department"].inverse_transform(dept_salary["Department"])
    st.bar_chart(dept_salary.set_index("Department")["Salary_INR"])

with col2:
    # Attrition rate per department
    dept_attr = df.groupby("Department")["Attrition"].mean().reset_index()
    dept_attr["Department"] = label_encoders["Department"].inverse_transform(dept_attr["Department"])
    st.bar_chart(dept_attr.set_index("Department")["Attrition"])
