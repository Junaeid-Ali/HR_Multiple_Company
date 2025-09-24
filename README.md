# HR_Multiple_Company
# 💼 HR Analytics & Prediction Project

## Overview
This project analyzes employee data from a multinational company and builds machine learning models to predict **employee attrition** and **salary**. Additionally, a **web-based dashboard** is created using Streamlit for interactive predictions and HR insights.

The project consists of:

1. **Exploratory Data Analysis (EDA)** on employee dataset.
2. **Data Preprocessing** including encoding categorical variables and scaling numerical features.
3. **Machine Learning Models**:
   - Classification: Predict employee attrition (Resigned/Terminated vs Active).
   - Regression: Predict employee salary.
4. **Web Dashboard**:
   - User inputs employee details to get predictions.
   - Interactive charts for HR insights.

---

## Dataset
**HR_Data_MNC_Data Science Lovers.csv**

Contains 30,000+ employee records with the following features:

| Column                  | Description |
|-------------------------|-------------|
| Employee_ID             | Unique identifier |
| Full_Name               | Employee name |
| Department              | Department of work |
| Job_Title               | Job designation |
| Hire_Date               | Date of hiring |
| Location                | City and country |
| Performance_Rating      | Numeric rating of performance |
| Experience_Years        | Years of experience |
| Status                  | Current status (Active, Resigned, Terminated, Retired) |
| Work_Mode               | On-site, Remote, or Hybrid |
| Salary_INR              | Annual salary in INR |

---

## Questions Answered (EDA)
The project answers multiple HR questions:

1. Distribution of Employee Status (Active, Resigned, Retired, Terminated)
2. Distribution of Work Modes (On-site, Remote)
3. Number of employees in each department
4. Average salary by department
5. Job title with the highest average salary
6. Average salary by department and job title
7. Number of employees Resigned/Terminated in each department
8. Salary variation with years of experience
9. Average performance rating by department
10. Country with the highest employee concentration
11. Correlation between performance rating and salary
12. Number of hires over time (per year)
13. Salary comparison: Remote vs On-site
14. Top 10 employees with highest salary in each department
15. Departments with highest attrition rate

---

## Machine Learning Models

1. **Classification**:
   - **Goal:** Predict if an employee will leave or stay.
   - **Model:** RandomForestClassifier
   - **Target:** Attrition (1 if Resigned/Terminated, 0 otherwise)
   - **Features:** Department, Job Title, Work Mode, Location, Country, Experience, Performance, Hire Year

2. **Regression**:
   - **Goal:** Predict employee salary.
   - **Model:** RandomForestRegressor
   - **Target:** Salary_INR
   - **Features:** Same as classification

---

## Streamlit Web Dashboard

The dashboard allows a user to:

- Enter employee details via dropdowns and numeric inputs
- Get **attrition prediction** with ✅ (Will Stay) or ❌ (Will Leave)
- Get **predicted salary** in INR
- Explore summary charts:
  - Average salary per department
  - Attrition rate per department

---

## Installation & Usage

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/HR-Prediction-Dashboard.git
cd HR-Prediction-Dashboard
