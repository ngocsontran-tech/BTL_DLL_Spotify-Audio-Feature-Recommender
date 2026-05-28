import os
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

def main():
    print("--- Phase 2: Data Preprocessing & Feature Engineering ---")

    # 1. Connect to MongoDB and load data
    print("Loading raw tracks from MongoDB...")
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["spotify_recommender"]
        raw_col = db["tracks"]
        
        raw_data = list(raw_col.find({}))
        if not raw_data:
            print("Error: No data found in 'tracks' collection. Please run data_collector.py first.")
            sys.exit(1)
            
        df = pd.DataFrame(raw_data)
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])
        print(f"Loaded {len(df)} tracks from MongoDB.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. Data Cleaning
    print("\n--- Cleaning Data ---")
    
    # 2.1 Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=['track_id'])
    print(f"Removed {initial_len - len(df)} duplicates. Remaining: {len(df)}")

    # 2.2 Handle missing values
    # List of numeric audio features
    numeric_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'popularity'
    ]
    
    # Convert columns to numeric types
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Explode genres list to calculate genre-specific medians
    df_exploded = df.explode('genres')
    genre_medians = df_exploded.groupby('genres')[numeric_cols].median()
    global_medians = df[numeric_cols].median()

    # Fill missing values
    filled_counts = {col: 0 for col in numeric_cols}
    for col in numeric_cols:
        null_mask = df[col].isnull()
        if null_mask.any():
            filled_counts[col] = null_mask.sum()
            
            def fill_val(row):
                genres_list = row['genres']
                if isinstance(genres_list, list) and len(genres_list) > 0:
                    first_genre = genres_list[0]
                    if first_genre in genre_medians.index and not pd.isna(genre_medians.loc[first_genre, col]):
                        return genre_medians.loc[first_genre, col]
                return global_medians[col]
                
            df.loc[null_mask, col] = df[null_mask].apply(fill_val, axis=1)
            
    print(f"Filled missing values: {filled_counts}")

    # 2.3 Handle outliers in tempo
    # Outliers defined as tempo < 40 or > 220 BPM
    tempo_median = df['tempo'].median()
    tempo_outliers_mask = (df['tempo'] < 40) | (df['tempo'] > 220)
    outliers_count = tempo_outliers_mask.sum()
    df.loc[tempo_outliers_mask, 'tempo'] = tempo_median
    print(f"Replaced {outliers_count} tempo outliers (<40 or >220 BPM) with median tempo ({tempo_median:.2f} BPM).")

    # 2.4 Normalize loudness from dB [-60, 0] to range [0, 1]
    df['loudness'] = df['loudness'].clip(-60.0, 0.0)
    df['loudness_norm'] = (df['loudness'] - (-60.0)) / (0.0 - (-60.0))
    print("Normalized loudness to [0, 1].")

    # 3. Feature Engineering
    print("\n--- Feature Engineering ---")

    # 3.1 Normalize tempo to range [0, 1] (tempo typically is between 40 and 220 BPM after cleaning)
    df['tempo_norm'] = (df['tempo'] - 40.0) / (220.0 - 40.0)
    df['tempo_norm'] = df['tempo_norm'].clip(0.0, 1.0)

    # 3.2 Ratio between energy and valence
    df['energy_valence_ratio'] = df['energy'] / (df['valence'] + 0.001)

    # 3.3 Combine valence, danceability, and energy into a single mood score
    df['mood_score'] = df['valence'] * 0.4 + df['danceability'] * 0.3 + df['energy'] * 0.3

    # 3.4 Categorize tempo: Slow (<90), Medium (90-130), Fast (>130)
    def categorize_tempo(t):
        if t < 90:
            return "Slow"
        elif t <= 130:
            return "Medium"
        else:
            return "Fast"
    df['tempo_category'] = df['tempo'].apply(categorize_tempo)

    # 3.5 Encode key and mode into one-hot columns
    # Save the original key and mode columns before encoding so they remain in the dataframe
    df['original_key'] = df['key']
    df['original_mode'] = df['mode']
    
    # Perform one-hot encoding on key (0 to 11) and mode (0, 1)
    df_processed = pd.get_dummies(df, columns=['key', 'mode'], prefix=['key', 'mode'], dtype=int)
    
    # Rename back key/mode columns to keep names tidy and clean
    df_processed = df_processed.rename(columns={'original_key': 'key', 'original_mode': 'mode'})

    # 4. Save Processed Data
    print("\n--- Saving Cleaned Data ---")
    
    # Save to MongoDB collection 'processed_tracks'
    processed_col = db["processed_tracks"]
    processed_col.drop() # Reset collection to avoid duplicates on re-runs
    processed_col.create_index("track_id", unique=True)
    
    # Drop the original MongoDB '_id' to avoid insert conflicts
    mongo_df = df_processed.copy()
    if '_id' in mongo_df.columns:
        mongo_df = mongo_df.drop(columns=['_id'])
        
    records = mongo_df.to_dict(orient='records')
    processed_col.insert_many(records)
    print(f"Saved {len(records)} cleaned tracks to MongoDB collection 'processed_tracks'.")

    # Save to Parquet file
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    parquet_path = os.path.join(processed_dir, "tracks_clean.parquet")
    df_processed.to_parquet(parquet_path, index=False)
    print(f"Saved cleaned tracks to Parquet file at: {parquet_path}")

    # 5. Output Statistics
    print("\n=================== STATISTICS REPORT ===================")
    
    # 5.1 Correlation Matrix of the 13 audio features
    features_for_corr = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'popularity'
    ]
    corr_matrix = df[features_for_corr].corr()
    print("Correlation Matrix (subset):")
    print(corr_matrix.loc[['danceability', 'energy', 'acousticness', 'valence'], ['danceability', 'energy', 'acousticness', 'valence']])
    
    # 5.2 Top 10 tracks by popularity
    print("\nTop 10 Most Popular Tracks:")
    top_10 = df_processed.sort_values(by='popularity', ascending=False).head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f" {i}. {row['track_name']} - {row['artist_name']} (Popularity: {row['popularity']})")

    # 5.3 Mood score distribution
    print("\nMood Score Distribution:")
    mood_desc = df_processed['mood_score'].describe()
    print(f" - Mean:   {mood_desc['mean']:.4f}")
    print(f" - Min:    {mood_desc['min']:.4f}")
    print(f" - 25%:    {mood_desc['25%']:.4f}")
    print(f" - Median: {mood_desc['50%']:.4f}")
    print(f" - 75%:    {mood_desc['75%']:.4f}")
    print(f" - Max:    {mood_desc['max']:.4f}")
    print("=========================================================")

if __name__ == "__main__":
    main()
