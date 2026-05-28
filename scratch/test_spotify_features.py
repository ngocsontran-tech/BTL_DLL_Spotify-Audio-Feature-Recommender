import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()
auth_manager = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Try fetching audio features for: "Vì Yêu Cứ Đâm Đầu" by MIN (id: 6j3m5TscV9c41o21w1Xq7P)
try:
    results = sp.audio_features(["6j3m5TscV9c41o21w1Xq7P"])
    print("Success! Audio features works.")
    print(results)
except Exception as e:
    print(f"Error: {e}")
