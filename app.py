from flask import Flask, render_template, request
import joblib
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Spotify API setup
sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyOAuth(
    client_id="fa7ba6e683f74682a32e1281fc0552dc",
    client_secret="7897a60bec414311964c753f97db25b7",
    redirect_uri="http://localhost:5001/callback",
    scope="user-library-read"
))


# List of required features
FEATURES = ['instrumentalness', 'danceability', 'loudness', 'valence', 'acousticness', 'key']

def get_song_features(song_name):
    """Fetch song features from Spotify API."""
    results = sp.search(q=song_name, type="track", limit=1)
    if not results['tracks']['items']:
        return None

    track_id = results['tracks']['items'][0]['id']
    audio_features = sp.audio_features(track_id)[0]

    if not audio_features:
        return None

    return [audio_features[feature] for feature in FEATURES]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    song_name = ""

    if request.method == "POST":
        song_name = request.form["song_name"]
        features = get_song_features(song_name)

        if features:
            features = np.array(features).reshape(1, -1)
            features = scaler.transform(features)
            prediction = model.predict(features)[0]
            prediction = "Hit" if prediction == 1 else "Flop"
        else:
            prediction = "Song not found!"

    return render_template("index.html", prediction=prediction, song_name=song_name)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
