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

try:
    results = sp.search(q="V-Pop", type="track", limit=5)
    print("Success! Search works.")
    for item in results['tracks']['items']:
        print(f"{item['name']} by {item['artists'][0]['name']}")
except Exception as e:
    print(f"Error: {e}")
