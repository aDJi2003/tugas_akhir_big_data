import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

data = pd.read_excel('data_pegawai.xlsx')

X = data.drop(columns=['Employee Intention', 'Name'])
y = data['Employee Intention']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

categorical_columns = ['Sex', 'Marital Status', 'Education']
numerical_columns = ['Years Employed', 'Work Hours', 'Career Development', 'Salary', 'Bonuses']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

def predict_employee_intention():
    print("\n=== Input Employee Details ===")
    name = input("Name: ")
    sex = input("Sex (M/F): ")
    marital_status = input("Marital Status (M/S/W): ")
    children = int(input("Number of Children: "))
    education = input("Education (College/Graduate Degree/Technical School/High School): ")
    years_employed = float(input("Years Employed: "))
    work_hours = float(input("Work Hours: "))
    career_development = int(input("Career Development Score: "))
    salary = float(input("Salary: "))
    bonuses = float(input("Bonuses: "))
    
    new_data = pd.DataFrame([{
        'Sex': sex,
        'Marital Status': marital_status,
        'Children': children,
        'Education': education,
        'Years Employed': years_employed,
        'Work Hours': work_hours,
        'Career Development': career_development,
        'Salary': salary,
        'Bonuses': bonuses
    }])

    prediction = pipeline.predict(new_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]
    
    print(f"\nPrediction for {name}: {prediction_label}")

predict_employee_intention()
