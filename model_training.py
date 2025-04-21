# train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv(r"D:\Project 1\Credit_score_classification\Dataset\train.csv")

# Drop unnecessary columns
columns_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Payment_Behaviour', 'Month']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Clean numeric columns
cols_to_numeric = ['Age', 'Annual_Income', 'Num_of_Loan', 'Changed_Credit_Limit', 'Outstanding_Debt', 'Credit_History_Age']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df[cols_to_numeric] = df[cols_to_numeric].fillna(df[cols_to_numeric].median())

# Fill object columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
if 'Credit_Mix' in df.columns:
    df['Credit_Mix'] = df['Credit_Mix'].map({'Bad': 0, 'Standard': 1, 'Good': 2}).fillna(1)

if 'Payment_of_Min_Amount' in df.columns:
    df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].map({'No': 0, 'Yes': 1, 'NM': 2}).fillna(2)

# Encode target
if 'Credit_Score' in df.columns:
    le = LabelEncoder()
    df['Credit_Score'] = le.fit_transform(df['Credit_Score'])

# Encode remaining object features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Get dummies
df = pd.get_dummies(df, drop_first=True)

# Split data  
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and columns
with open("credit_score_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("✅ Model and columns saved.")
