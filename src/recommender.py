import os
import sys
import re
import hashlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor


# Load environment variables
load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

FEATURE_COLS = [
    'danceability', 'energy', 'acousticness', 'instrumentalness',
    'valence', 'tempo_norm', 'speechiness', 'liveness', 'loudness_norm', 'mood_score'
]

# Initialize MongoDB
try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client["spotify_recommender"]
    processed_col = db["processed_tracks"]
except Exception as e:
    print(f"Error connecting to MongoDB in recommender: {e}")
    sys.exit(1)

# Initialize Spotipy
try:
    auth_manager = SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(auth_manager=auth_manager, retries=0, status_retries=0, status_forcelist=(999,))
except Exception as e:
    print(f"Error authenticating with Spotify in recommender: {e}")
    sp = None

import time

def spotify_api_call(func, *args, **kwargs):
    """
    Wrapper for Spotify API calls. Handles rate limit (429) by reading
    'Retry-After' header, sleeping for the requested duration, and retrying.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                retry_after = 5
                if e.headers:
                    for k, v in e.headers.items():
                        if k.lower() == 'retry-after':
                            try:
                                retry_after = int(v)
                                break
                            except ValueError:
                                pass
                if retry_after > 10:
                    print(f"Spotify API rate limit retry-after is too large ({retry_after}s). Raising exception immediately.")
                    raise e
                print(f"Spotify API 429 Rate Limit. Sleeping for {retry_after} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(retry_after)
                continue
            raise e
        except Exception as e:
            raise e

def parse_track_id(url):
    """
    Parses a track_id from a Spotify URL, URI, or returns it if already an ID.
    Supports track URLs, artist URLs (resolves to their most popular track via Search),
    and album URLs (resolves to the first track of the album).
    """
    if not url:
        raise ValueError("URL cannot be empty.")
    url = url.strip()
    
    # 1. Already a track ID
    if len(url) == 22 and url.isalnum():
        return url
        
    # 2. Track URL/URI
    if url.startswith("spotify:track:"):
        return url.split(":")[-1]
    match_track = re.search(r"track/([a-zA-Z0-9]+)", url)
    if match_track:
        return match_track.group(1)
        
    # 3. Artist URL/URI -> Resolve to top track
    artist_id = None
    if url.startswith("spotify:artist:"):
        artist_id = url.split(":")[-1]
    else:
        match_artist = re.search(r"artist/([a-zA-Z0-9]+)", url)
        if match_artist:
            artist_id = match_artist.group(1)
            
    if artist_id:
        if not sp:
            raise ValueError("Spotify API not initialized. Cannot resolve artist URL.")
        try:
            artist_info = spotify_api_call(sp.artist, artist_id)
            if artist_info is None:
                raise ValueError(f"Could not fetch artist info for artist ID: {artist_id}")
            artist_name = artist_info['name']
            search_results = spotify_api_call(sp.search, q=f'artist:"{artist_name}"', type='track', limit=1)
            if search_results and 'tracks' in search_results and search_results['tracks']['items']:
                track = search_results['tracks']['items'][0]
                print(f"Resolved artist URL to their top track: '{track['name']}' by {artist_name} (ID: {track['id']})")
                return track['id']
            else:
                raise ValueError(f"Could not find any tracks for artist: {artist_name}")
        except Exception as e:
            raise ValueError(f"Failed to resolve artist URL: {e}")
            
    # 4. Album URL/URI -> Resolve to first track
    album_id = None
    if url.startswith("spotify:album:"):
        album_id = url.split(":")[-1]
    else:
        match_album = re.search(r"album/([a-zA-Z0-9]+)", url)
        if match_album:
            album_id = match_album.group(1)
            
    if album_id:
        if not sp:
            raise ValueError("Spotify API not initialized. Cannot resolve album URL.")
        try:
            album_info = spotify_api_call(sp.album, album_id)
            if album_info and 'tracks' in album_info and album_info['tracks']['items']:
                track = album_info['tracks']['items'][0]
                print(f"Resolved album URL to its first track: '{track['name']}' (ID: {track['id']})")
                return track['id']
            else:
                album_name = album_info.get('name', album_id) if album_info is not None else album_id
                raise ValueError(f"Could not find any tracks in album: {album_name}")
        except Exception as e:
            # Check if rate limit
            err_str = str(e)
            if "rate/request limit" in err_str.lower() or "429" in err_str:
                raise ValueError("Spotify API Rate Limit: Không thể phân tích album do đạt giới hạn yêu cầu từ Spotify.")
            raise ValueError(f"Failed to resolve album URL: {e}")

    # 5. Playlist URLs -> User friendly error
    if "playlist" in url or "spotify:playlist:" in url:
        raise ValueError("Spotify playlist URLs are not supported due to API restrictions. Please use a track, artist, or album URL.")
        
    raise ValueError(f"Could not parse Spotify track, artist, or album ID from URL: {url}")


def generate_mock_features(track_id):
    """
    Generates deterministic mock features based on the hash of track_id
    as a fallback for Spotify's deprecated audio-features endpoint.
    """
    h = hashlib.md5(track_id.encode('utf-8')).hexdigest()
    def get_val(start_idx, end_idx, min_val=0.0, max_val=1.0):
        val = int(h[start_idx:end_idx], 16) / (16**(end_idx-start_idx))
        return min_val + val * (max_val - min_val)

    danceability = get_val(0, 4)
    energy = get_val(4, 8)
    acousticness = get_val(8, 12)
    instrumentalness = get_val(12, 16, max_val=0.2)
    valence = get_val(16, 20)
    tempo_norm = get_val(20, 24)
    speechiness = get_val(24, 26, max_val=0.3)
    liveness = get_val(26, 28, max_val=0.4)
    loudness_norm = get_val(28, 30, min_val=0.4, max_val=0.9)
    mood_score = valence * 0.4 + danceability * 0.3 + energy * 0.3
    
    return {
        "danceability": danceability,
        "energy": energy,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "valence": valence,
        "tempo_norm": tempo_norm,
        "speechiness": speechiness,
        "liveness": liveness,
        "loudness_norm": loudness_norm,
        "mood_score": mood_score
    }

def get_track_metadata_fallback(track_id):
    """
    Queries Spotify API for the track ID to retrieve real metadata.
    First tries sp.track(), then falls back to sp.search(), and finally a generic fallback.
    """
    if sp:
        # Try direct track lookup first (most reliable)
        try:
            track = spotify_api_call(sp.track, track_id)
            if track:
                return {
                    "track_name": track['name'],
                    "artist_name": ", ".join([a['name'] for a in track['artists']]),
                    "album_name": track['album']['name'],
                    "release_year": int(track['album']['release_date'].split("-")[0]) if track['album']['release_date'] else None,
                    "popularity": track.get('popularity', 50),
                    "album_cover_url": track['album']['images'][0]['url'] if track['album']['images'] else None,
                    "spotify_url": f"https://open.spotify.com/track/{track_id}"
                }
        except Exception as e:
            print(f"Warning: Direct track lookup failed for {track_id}: {e}")

        # Fallback to search query
        try:
            results = spotify_api_call(sp.search, q=track_id, type="track", limit=1)
            if results and 'tracks' in results and 'items' in results['tracks'] and results['tracks']['items']:
                track = results['tracks']['items'][0]
                if track['id'] == track_id:
                    return {
                        "track_name": track['name'],
                        "artist_name": ", ".join([a['name'] for a in track['artists']]),
                        "album_name": track['album']['name'],
                        "release_year": int(track['album']['release_date'].split("-")[0]) if track['album']['release_date'] else None,
                        "popularity": track.get('popularity', 50),
                        "album_cover_url": track['album']['images'][0]['url'] if track['album']['images'] else None,
                        "spotify_url": f"https://open.spotify.com/track/{track_id}"
                    }
        except Exception as e:
            print(f"Warning: Could not fetch track metadata via search: {e}")
            
    # Generic fallback if search fails or is unauthorized
    return {
        "track_name": f"Track {track_id[:8]}",
        "artist_name": "Unknown Artist",
        "album_name": "Unknown Album",
        "release_year": None,
        "popularity": 50,
        "album_cover_url": "data/covers/cover_synthwave.png",
        "spotify_url": f"https://open.spotify.com/track/{track_id}"
    }


def get_centroids():
    """
    Calculates the centroids of the 6 clusters on the fly from MongoDB.
    """
    pipeline = [
        {"$group": {
            "_id": "$cluster",
            "danceability": {"$avg": "$danceability"},
            "energy": {"$avg": "$energy"},
            "acousticness": {"$avg": "$acousticness"},
            "instrumentalness": {"$avg": "$instrumentalness"},
            "valence": {"$avg": "$valence"},
            "tempo_norm": {"$avg": "$tempo_norm"},
            "speechiness": {"$avg": "$speechiness"},
            "liveness": {"$avg": "$liveness"},
            "loudness_norm": {"$avg": "$loudness_norm"},
            "mood_score": {"$avg": "$mood_score"}
        }}
    ]
    centroids = list(processed_col.aggregate(pipeline))
    return {c['_id']: c for c in centroids}

def assign_cluster_to_features(features, centroids):
    """
    Assigns a cluster ID to a feature dict based on minimum Euclidean distance to centroids.
    """
    min_dist = float('inf')
    best_cluster = 0
    for c_id, c in centroids.items():
        dist = 0
        for feat in FEATURE_COLS:
            dist += (features[feat] - c[feat]) ** 2
        dist = dist ** 0.5
        if dist < min_dist:
            min_dist = dist
            best_cluster = c_id
    return best_cluster

def get_track_features(spotify_url):
    """
    Fetches the features and metadata of a track from MongoDB or fallbacks safely.
    """
    track_id = parse_track_id(spotify_url)
    
    # 1. Try to find track in MongoDB
    track_doc = processed_col.find_one({"track_id": track_id})
    if track_doc:
        # If it has a mock cover, fetch the real cover dynamically from Spotify API
        cover = track_doc.get('album_cover_url', '')
        if not cover or cover.startswith('data/covers/'):
            try:
                real_meta = get_track_metadata_fallback(track_id)
                real_cover = real_meta.get('album_cover_url')
                if real_cover and not real_cover.startswith('data/covers/'):
                    track_doc['album_cover_url'] = real_cover
                    # Update MongoDB so we cache this forever!
                    processed_col.update_one(
                        {"track_id": track_id},
                        {"$set": {"album_cover_url": real_cover}}
                    )
            except Exception:
                pass
        return track_doc

    # 2. If not in DB, retrieve real metadata and generate deterministic mock features
    print(f"Track {track_id} not found in DB. Fetching metadata and generating features...")
    metadata = get_track_metadata_fallback(track_id)
    features = generate_mock_features(track_id)
    
    # Calculate cluster
    centroids = get_centroids()
    cluster_id = assign_cluster_to_features(features, centroids)
    
    track_info = {
        "track_id": track_id,
        "track_name": metadata["track_name"],
        "artist_name": metadata["artist_name"],
        "album_name": metadata["album_name"],
        "release_year": metadata["release_year"],
        "popularity": metadata["popularity"],
        "album_cover_url": metadata["album_cover_url"],
        "spotify_url": metadata["spotify_url"],
        "cluster": cluster_id,
        **features
    }
    
    return track_info

def cosine_similarity(v1, v2):
    """
    Calculates cosine similarity between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def recommend_songs(spotify_url, top_n=10):
    """
    Recommends top_n similar songs from the same cluster based on cosine similarity.
    """
    # Step 1: Get input track features
    input_track = get_track_features(spotify_url)
    input_cluster = input_track["cluster"]
    input_vector = np.array([input_track[col] for col in FEATURE_COLS])

    print(f"\nSearching recommendations for track:")
    print(f" - Title: {input_track['track_name']} by {input_track['artist_name']}")
    print(f" - Cluster: {input_cluster} | Popularity: {input_track['popularity']}")
    
    # Step 2: Fetch all database tracks from the SAME cluster (reduces search space)
    candidate_tracks = list(processed_col.find({"cluster": input_cluster}))
    print(f"Found {len(candidate_tracks)} tracks in Cluster {input_cluster} to search from.")

    # Step 3: Compute Cosine Similarity
    recommendations = []
    for track in candidate_tracks:
        # Exclude the input track itself
        if track["track_id"] == input_track["track_id"]:
            continue
            
        track_vector = np.array([track[col] for col in FEATURE_COLS])
        sim = cosine_similarity(input_vector, track_vector)
        
        recommendations.append({
            "track_id": track["track_id"],
            "track_name": track["track_name"],
            "artist_name": track["artist_name"],
            "album_name": track["album_name"],
            "popularity": track["popularity"],
            "album_cover_url": track["album_cover_url"],
            "spotify_url": track["spotify_url"],
            "similarity": sim
        })

    # Step 4: Sort by similarity in descending order
    recommendations.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top_n
    return recommendations[:top_n]

def fetch_real_covers(tracks_list):
    """
    Fetches real album cover URLs from Spotify API for a list of tracks in batch,
    updating the 'album_cover_url' field in place and caching them in MongoDB.
    If the batch endpoint (sp.tracks) is forbidden, falls back to individual calls in parallel.
    """
    if not sp:
        return tracks_list
        
    to_fetch = []
    for track in tracks_list:
        cover = track.get('album_cover_url', '')
        if not cover or cover.startswith('data/covers/'):
            to_fetch.append(track)
            
    if not to_fetch:
        return tracks_list
        
    chunk_size = 50
    for i in range(0, len(to_fetch), chunk_size):
        chunk = to_fetch[i:i + chunk_size]
        track_ids = [t['track_id'] for t in chunk]
        
        try:
            response = spotify_api_call(sp.tracks, track_ids)
            if response and 'tracks' in response:
                for track_dict, api_track in zip(chunk, response['tracks']):
                    if api_track and 'album' in api_track and api_track['album'].get('images'):
                        real_url = api_track['album']['images'][0]['url']
                        track_dict['album_cover_url'] = real_url
                        # Cache in MongoDB
                        processed_col.update_one(
                            {"track_id": track_dict['track_id']},
                            {"$set": {"album_cover_url": real_url}}
                        )
        except Exception as e:
            print(f"Warning: Failed to fetch real covers in batch ({e}). Falling back to individual parallel fetching...")
            
            # Helper function for individual fetching
            def fetch_single_cover(track_dict):
                try:
                    track_info = spotify_api_call(sp.track, track_dict['track_id'])
                    if track_info and 'album' in track_info and track_info['album'].get('images'):
                        real_url = track_info['album']['images'][0]['url']
                        track_dict['album_cover_url'] = real_url
                        # Cache in MongoDB
                        processed_col.update_one(
                            {"track_id": track_dict['track_id']},
                            {"$set": {"album_cover_url": real_url}}
                        )
                except Exception as single_err:
                    print(f"Warning: Failed to fetch cover for track {track_dict['track_id']}: {single_err}")

            # Execute in parallel threads to keep it extremely fast
            with ThreadPoolExecutor(max_workers=10) as executor:
                executor.map(fetch_single_cover, chunk)
            
    return tracks_list

if __name__ == "__main__":
    print("--- Phase 4: Recommendation Engine Test ---")
    
    # We will test using "Blinding Lights" by The Weeknd (track_id: 0SF9q7EX3u236q6P2AqGJu)
    # This track is a mega pop hit present in our database
    test_url = "https://open.spotify.com/track/0sf12qNH5qcw8qpgymFOqD"
    
    try:
        recs = recommend_songs(test_url, top_n=10)
        print("\n=================== TOP 10 RECOMMENDATIONS ===================")
        for idx, rec in enumerate(recs, 1):
            print(f" {idx:2d}. {rec['track_name']} - {rec['artist_name']} (Similarity: {rec['similarity']:.4%}, Pop: {rec['popularity']})")
        print("==============================================================")
    except Exception as e:
        print(f"Error running recommender test: {e}")
