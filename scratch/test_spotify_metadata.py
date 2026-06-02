# type: ignore
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

# Test sp.tracks on 6f807x0ima9a1j3VPbc7VN
try:
    results = sp.tracks(["6f807x0ima9a1j3VPbc7VN"])
    print("Success! Track metadata works.")
    if results is not None:
        track = results['tracks'][0]
        print(f"Name: {track['name']}")
        print(f"Cover URL: {track['album']['images'][0]['url']}")
except Exception as e:
    print(f"Error: {e}")
