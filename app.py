from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = ['instrumentalness', 'danceability', 'loudness', 'valence', 'acousticness', 'key']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    feature_values = {}

    if request.method == "POST":
        try:
            feature_values = {feature: float(request.form[feature]) for feature in FEATURES}

            features_array = np.array([list(feature_values.values())]).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            prediction = model.predict(features_scaled)[0]
            prediction = "Hit 🎵" if prediction == 1 else "Flop 💔"
        except ValueError:
            prediction = "Invalid input! Please enter numeric values."

    return render_template("index.html", prediction=prediction, feature_values=feature_values)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
