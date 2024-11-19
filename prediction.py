import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
@st.cache
def load_data():
    df = pd.read_csv(r'C:\Users\Administrator\Downloads\diabetes.csv')  # Ensure the file is in the same directory or provide the correct path
    return df

df = load_data()

# Title and Description
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Split Data into X and Y
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Function to Collect User Data
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    Skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.0, 20.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': Skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# User Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Model Training
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# User Prediction
user_result = rf.predict(user_data)

# Visualizations
st.title('Visualized Patient Report')

# Helper Function for Color
def get_color(prediction):
    return 'blue' if prediction == 0 else 'red'

# Color for the current user
color = get_color(user_result[0])

# Plotting Graphs
def plot_graph(feature_x, feature_y, user_feature_x, user_feature_y, title, palette):
    fig = plt.figure()
    sns.scatterplot(x=feature_x, y=feature_y, data=df, hue='Outcome', palette=palette)
    sns.scatterplot(x=user_feature_x, y=user_feature_y, s=150, color=color)
    plt.title(title)
    st.pyplot(fig)

# Generate Visualizations
plot_graph('Age', 'Pregnancies', user_data['Age'], user_data['Pregnancies'], 
           'Pregnancy Count Graph (Others vs Yours)', 'Greens')
plot_graph('Age', 'Glucose', user_data['Age'], user_data['Glucose'], 
           'Glucose Value Graph (Others vs Yours)', 'magma')
plot_graph('Age', 'BloodPressure', user_data['Age'], user_data['BloodPressure'], 
           'Blood Pressure Graph (Others vs Yours)', 'Reds')
plot_graph('Age', 'SkinThickness', user_data['Age'], user_data['SkinThickness'], 
           'Skin Thickness Graph (Others vs Yours)', 'Blues')
plot_graph('Age', 'Insulin', user_data['Age'], user_data['Insulin'], 
           'Insulin Graph (Others vs Yours)', 'rocket')
plot_graph('Age', 'BMI', user_data['Age'], user_data['BMI'], 
           'BMI Graph (Others vs Yours)', 'rainbow')
plot_graph('Age', 'DiabetesPedigreeFunction', user_data['Age'], user_data['DiabetesPedigreeFunction'], 
           'Diabetes Pedigree Function Graph (Others vs Yours)', 'YlOrBr')

# Output Results
st.subheader('Your Report:')
if user_result[0] == 0:
    st.title('You are not Diabetic')
else:
    st.title('You are Diabetic')

# Accuracy
st.subheader('Model Accuracy:')
st.write(f"{accuracy_score(y_test, rf.predict(X_test)) * 100:.2f}%")
