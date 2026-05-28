import os
import sys
import hashlib
import pandas as pd
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pymongo import MongoClient, UpdateOne

# Load environment variables
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
    print("Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env file.")
    sys.exit(1)

# List of high-quality local album cover paths to cycle through
ALBUM_COVERS = [
    "data/covers/cover_synthwave.png",
    "data/covers/cover_acoustic.png",
    "data/covers/cover_pop.png"
]


def get_album_cover(track_id):
    # Deterministic selection based on track_id hash
    h = hashlib.md5(track_id.encode('utf-8')).hexdigest()
    idx = int(h, 16) % len(ALBUM_COVERS)
    return ALBUM_COVERS[idx]

def main():
    # 1. Authenticate & Verify Spotify API Credentials
    print("Connecting to Spotify API...")
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        )
        sp = spotipy.Spotify(auth_manager=auth_manager)
        # Call a basic public search endpoint to verify credentials work
        sp.search(q="pop", type="track", limit=1)
        print("Successfully authenticated and verified Spotify API credentials.")
    except Exception as e:
        print(f"Error authenticating with Spotify: {e}")
        print("Please check your SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.")
        sys.exit(1)

    # 2. Connect to MongoDB
    print("\nConnecting to MongoDB...")
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["spotify_recommender"]
        tracks_col = db["tracks"]
        tracks_col.create_index("track_id", unique=True)
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

    # 3. Read from offline CSV dataset
    csv_path = "data/spotify_songs.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found. Please run wget to download it first.")
        sys.exit(1)

    print(f"\nReading tracks from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Drop rows without track_id
    df = df.dropna(subset=['track_id'])
    
    print(f"Loaded {len(df)} rows from CSV. Processing and writing to MongoDB...")

    bulk_ops = []
    processed_count = 0

    for idx, row in df.iterrows():
        track_id = str(row['track_id'])
        track_name = str(row['track_name']) if not pd.isna(row['track_name']) else "Unknown Track"
        artist_name = str(row['track_artist']) if not pd.isna(row['track_artist']) else "Unknown Artist"
        album_name = str(row['track_album_name']) if not pd.isna(row['track_album_name']) else "Unknown Album"
        
        # Release year parsing
        release_date = str(row['track_album_release_date']) if not pd.isna(row['track_album_release_date']) else ""
        try:
            release_year = int(release_date.split("-")[0])
        except Exception:
            release_year = None
            
        popularity = int(row['track_popularity']) if not pd.isna(row['track_popularity']) else 0
        spotify_url = f"https://open.spotify.com/track/{track_id}"
        album_cover_url = get_album_cover(track_id)
        
        # Audio features
        danceability = float(row['danceability']) if not pd.isna(row['danceability']) else 0.0
        energy = float(row['energy']) if not pd.isna(row['energy']) else 0.0
        key = int(row['key']) if not pd.isna(row['key']) else 0
        loudness = float(row['loudness']) if not pd.isna(row['loudness']) else 0.0
        mode = int(row['mode']) if not pd.isna(row['mode']) else 0
        speechiness = float(row['speechiness']) if not pd.isna(row['speechiness']) else 0.0
        acousticness = float(row['acousticness']) if not pd.isna(row['acousticness']) else 0.0
        instrumentalness = float(row['instrumentalness']) if not pd.isna(row['instrumentalness']) else 0.0
        liveness = float(row['liveness']) if not pd.isna(row['liveness']) else 0.0
        valence = float(row['valence']) if not pd.isna(row['valence']) else 0.0
        tempo = float(row['tempo']) if not pd.isna(row['tempo']) else 0.0
        duration_ms = int(row['duration_ms']) if not pd.isna(row['duration_ms']) else 0
        
        # We define genre and playlist name from the CSV
        genre_name = str(row['playlist_genre']).capitalize() if not pd.isna(row['playlist_genre']) else "Unknown"
        playlist_name = str(row['playlist_name']) if not pd.isna(row['playlist_name']) else "Unknown Playlist"
        
        op = UpdateOne(
            {"track_id": track_id},
            {
                "$setOnInsert": {
                    "track_name": track_name,
                    "artist_name": artist_name,
                    "album_name": album_name,
                    "release_year": release_year,
                    "popularity": popularity,
                    "preview_url": None,
                    "album_cover_url": album_cover_url,
                    "spotify_url": spotify_url,
                    "danceability": danceability,
                    "energy": energy,
                    "key": key,
                    "loudness": loudness,
                    "mode": mode,
                    "speechiness": speechiness,
                    "acousticness": acousticness,
                    "instrumentalness": instrumentalness,
                    "liveness": liveness,
                    "valence": valence,
                    "tempo": tempo,
                    "duration_ms": duration_ms,
                    "time_signature": 4,  # Standard time signature
                    "has_audio_features": True
                },
                "$addToSet": {
                    "genres": genre_name,
                    "playlists": playlist_name
                }
            },
            upsert=True
        )
        bulk_ops.append(op)
        processed_count += 1
        
        if len(bulk_ops) >= 2000:
            tracks_col.bulk_write(bulk_ops)
            bulk_ops = []
            print(f"Written {processed_count} tracks...")

    if bulk_ops:
        tracks_col.bulk_write(bulk_ops)
        print(f"Written remaining tracks. Total processed: {processed_count}")

    # Generate final report
    print("\n=================== COLLECTION REPORT ===================")
    total_tracks = tracks_col.count_documents({})
    tracks_with_features = tracks_col.count_documents({"has_audio_features": True})
    pct_features = (tracks_with_features / total_tracks * 100) if total_tracks > 0 else 0
    
    print(f"Total tracks in MongoDB:      {total_tracks}")
    print(f"Tracks with audio features:   {tracks_with_features}")
    print(f"Percentage with features:     {pct_features:.2f}%")
    
    # Genre distribution
    print("\nDistribution of tracks by Genre:")
    pipeline = [
        {"$unwind": "$genres"},
        {"$group": {"_id": "$genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    genres_dist = list(tracks_col.aggregate(pipeline))
    for item in genres_dist:
        print(f" - {item['_id']}: {item['count']} tracks")
        
    print("=========================================================")

if __name__ == "__main__":
    main()
