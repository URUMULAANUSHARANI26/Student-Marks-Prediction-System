import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# 1. Load the updated dataset
data = pd.read_csv("student_data.csv")

# 2. Convert Gender text to numbers (Male = 1, Female = 0)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# 3. Features (Inputs) and Target (Output)
X = data[["Hours_Studied", "Attendance"]] # You can add , "Gender" here later if you want to train on it!
y = data["Marks"]

# 4. Train the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 5. Save the trained model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🚀 Model trained successfully with the updated dataset!")