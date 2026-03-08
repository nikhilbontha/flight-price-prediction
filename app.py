import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("✈️ Flight Price Prediction App")

# Step 1: Load Data directly (no upload)
#df = pd.read_csv("Data_Train.csv", encoding="latin1")
df = pd.read_excel("Data_Train.xlsx", engine="openpyxl")

st.subheader("Raw Data Preview")
st.write(df.head())

# Step 2: Data Cleaning
st.subheader("Data Info")
st.write("Missing Values:", df.isnull().sum())
df.dropna(inplace=True)

# Step 3: Feature Engineering
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], format='%d/%m/%Y')
df['day'] = df['Date_of_Journey'].dt.day
df['month'] = df['Date_of_Journey'].dt.month
df['year'] = df['Date_of_Journey'].dt.year
df.drop('Date_of_Journey', axis=1, inplace=True)

df['Dep_Time'] = pd.to_datetime(df['Dep_Time'])
df['Dep_hour'] = df['Dep_Time'].dt.hour
df['Dep_min'] = df['Dep_Time'].dt.minute
df.drop('Dep_Time', axis=1, inplace=True)

df['Arrival_Time'] = pd.to_datetime(df['Arrival_Time'])
df['Arrival_hour'] = df['Arrival_Time'].dt.hour
df['Arrival_min'] = df['Arrival_Time'].dt.minute

df['Duration'] = pd.to_timedelta(df['Duration'])
df['Duration_hour'] = df['Duration'].dt.components.hours
df['Duration_min'] = df['Duration'].dt.components.minutes

df.drop(['Arrival_Time', 'Duration', 'Additional_Info'], axis=1, inplace=True)

# Encode Total Stops
df['Total_Stops'].replace({
    'non-stop': 0, '1 stop': 1, '2 stops': 2,
    '3 stops': 3, '4 stops': 4
}, inplace=True)

# Route Feature Engineering
df['Route_count'] = df['Route'].apply(lambda x: len(x.split('→')))
df['Route_1'] = df['Route'].apply(lambda x: x.split('→')[0])
df['Route_2'] = df['Route'].apply(lambda x: x.split('→')[1] if len(x.split('→')) > 1 else 'None')
df['Route_3'] = df['Route'].apply(lambda x: x.split('→')[2] if len(x.split('→')) > 2 else 'None')
df['Route_4'] = df['Route'].apply(lambda x: x.split('→')[3] if len(x.split('→')) > 3 else 'None')
df['Route_5'] = df['Route'].apply(lambda x: x.split('→')[4] if len(x.split('→')) > 4 else 'None')
df.drop("Route", axis=1, inplace=True)

# Step 4: Visualization
st.subheader("Price vs Airline")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Airline', y='Price', data=df, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Step 5: Encoding Categorical Variables
le = LabelEncoder()
for col in ['Airline', 'Source', 'Destination', 'Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']:
    df[col] = le.fit_transform(df[col])

st.subheader("Processed Data Preview")
st.write(df.head())

# Step 6: Regression Model
st.subheader("Model Training")

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"✅ Model Trained: Linear Regression")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")

# Step 7: Prediction Form
st.subheader("Predict Flight Price")
airline = st.number_input("Airline (encoded)", min_value=0, max_value=df['Airline'].max(), value=0)
source = st.number_input("Source (encoded)", min_value=0, max_value=df['Source'].max(), value=0)
destination = st.number_input("Destination (encoded)", min_value=0, max_value=df['Destination'].max(), value=0)
total_stops = st.number_input("Total Stops", min_value=0, max_value=4, value=0)
dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23, value=10)
dep_min = st.number_input("Departure Minute", min_value=0, max_value=59, value=30)
duration_hour = st.number_input("Duration Hour", min_value=0, max_value=24, value=2)
duration_min = st.number_input("Duration Minute", min_value=0, max_value=59, value=30)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        'Airline':[airline],
        'Source':[source],
        'Destination':[destination],
        'Total_Stops':[total_stops],
        'Dep_hour':[dep_hour],
        'Dep_min':[dep_min],
        'Arrival_hour':[0],
        'Arrival_min':[0],
        'Duration_hour':[duration_hour],
        'Duration_min':[duration_min],
        'day':[1],
        'month':[1],
        'year':[2024],
        'Route_count':[1],
        'Route_1':[0],
        'Route_2':[0],
        'Route_3':[0],
        'Route_4':[0],
        'Route_5':[0]
    })

    # 🔑 Important line
    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Flight Price: ₹{prediction:,.2f}")