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
    'dataset-of-90s.csv', 'dataset-of-00s.csv', 'dataset-of-10s.csv', 'Merged file.csv'
]

# Selected features for training
selected_features = ['instrumentalness', 'danceability', 'loudness', 'valence', 'acousticness', 'key']

# Loop through each CSV file
for file in csv_files:
    print(f"Processing {file}...")
    df = pd.read_csv(file).drop(columns=['uri'], errors='ignore')
    
    # Splitting features and target
    X = df[selected_features]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Results for {file}: Random Forest Accuracy = {accuracy:.4f}\n")
    
    # Save the model and scaler
    joblib.dump(model, f'random_forest_model_{file.replace(".csv", "").replace(" ", "_")}.pkl')
    joblib.dump(scaler, f'scaler_{file.replace(".csv", "").replace(" ", "_")}.pkl')
