import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Create dataset
data = {
    'Income': ['High', 'Low', 'Medium', 'High',
               'Low', 'Medium'],
    'Credit': ['Good', 'Bad', 'Good', 'Good',
               'Bad', 'Good'],
    'Loan': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Convert text into numbers
le = LabelEncoder()

for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Features and target
X = df[['Income', 'Credit']]
y = df['Loan']

# Create model
model = GaussianNB()

# Train model
model.fit(X, y)

# Predict new loan applicant
result = model.predict([[0, 1]])

print("Loan Prediction:",
      "Yes" if result[0] == 1 else "No")