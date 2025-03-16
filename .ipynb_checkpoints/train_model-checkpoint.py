import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load CSV with correct column names  
columns = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean",  
           "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",  
           "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",  
           "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",  
           "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",  
           "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",  
           "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"]  

df = pd.read_csv("breast_cancer_data.csv", names=columns, header=None)  

# Convert labels to numeric  
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  

# Drop unnecessary columns  
df = df.drop(columns=["id"])  # 'id' is not needed for training  

# Split dataset  
X = df.drop(columns=["diagnosis"])  # Use all 30 features  
y = df["diagnosis"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model  
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy  
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

# Save the trained model  
model_filename = "breast_cancer_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)
print(f"✅ Model saved as '{model_filename}'")
