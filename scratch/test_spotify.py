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

# Try fetching a known global playlist: Top 50 - Global (37i9dQZEVXbMDoHDGih2jR)
try:
    playlist = sp.playlist_tracks("37i9dQZEVXbMDoHDGih2jR", limit=5)
    print("Success! Fetched playlist tracks.")
    for item in playlist['items']:
        print(item['track']['name'])
except Exception as e:
    print(f"Error fetching playlist tracks: {e}")
