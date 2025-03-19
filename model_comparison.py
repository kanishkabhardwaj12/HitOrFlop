import pandas as pd
import tensorflow as tf
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# List of CSV files
csv_files = ['dataset-of-60s.csv', 'dataset-of-70s.csv', 'dataset-of-80s.csv', 'dataset-of-90s.csv', 'dataset-of-00s.csv','dataset-of-10s.csv','Merged file.csv']

# Loop through each CSV file
for file in csv_files:
    print(f"Processing {file}...")
    df = pd.read_csv(file)
    df = df.drop(columns=['uri'])

    x = df.iloc[:, 2:-1]
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Deep learning model
    model4 = Sequential()
    #layers
    model4.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model4.add(Dense(32, activation='relu'))
    model4.add(Dense(1, activation='sigmoid'))

    model4.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    model4.fit(x_train, y_train, epochs=20, batch_size=22, validation_data=(x_test, y_test))

    loss, accuracy = model4.evaluate(x_test, y_test)

    # Traditional ML models
    model = LogisticRegression()
    model1 = DecisionTreeClassifier()
    model2 = RandomForestClassifier()
    model3 = SVC()

    model.fit(x_train, y_train)
    model1.fit(x_train, y_train)
    model2.fit(x_train, y_train)
    model3.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred1 = model1.predict(x_test)
    y_pred2 = model2.predict(x_test)
    y_pred3 = model3.predict(x_test)

    a = accuracy_score(y_test, y_pred)
    b = accuracy_score(y_test, y_pred1)
    c = accuracy_score(y_test, y_pred2)
    d = accuracy_score(y_test, y_pred3)

    print(f"Results for {file}:")
    print(f"Logistic Regression Accuracy: {a}")
    print(f"Decision Tree Accuracy: {b}")
    print(f"Random Forest Accuracy: {c}")
    print(f"SVM Accuracy: {d}")
    print(f"Neural Network Accuracy: {accuracy}")
    print("\n")
