import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

csv_files = [
    'dataset-of-60s.csv', 'dataset-of-70s.csv', 'dataset-of-80s.csv',
    'dataset-of-90s.csv', 'dataset-of-00s.csv', 'dataset-of-10s.csv'
]

selected_features = ['instrumentalness', 'danceability', 'loudness', 'valence', 'acousticness', 'key']

df_list = [pd.read_csv(file).drop(columns=['uri'], errors='ignore') for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_df.to_csv("merged_dataset.csv", index=False)
print("All CSV files have been merged into 'merged_dataset.csv'")

# Splitting features and target
X = merged_df[selected_features]
y = merged_df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy = {accuracy:.4f}")

# Save the model and scaler
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
