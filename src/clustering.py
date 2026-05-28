import os
import sys
from pymongo import MongoClient, UpdateOne

# Set PySpark env variables to use the virtual environment's python
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

def main():
    print("--- Phase 3: Gom cụm bằng Apache Spark (K-Means) ---")

    # 1. Initialize Spark Session
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName("SpotifyClustering") \
            .master("local[*]") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        print("Spark Session initialized successfully.")
    except Exception as e:
        print(f"Error initializing Spark: {e}")
        sys.exit(1)

    # 2. Read processed Parquet dataset
    parquet_path = "data/processed/tracks_clean.parquet"
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file {parquet_path} not found. Please run preprocessing.py first.")
        spark.stop()
        sys.exit(1)

    print(f"Reading cleaned data from Parquet file: {parquet_path}...")
    df_spark = spark.read.parquet(parquet_path)
    print(f"Loaded {df_spark.count()} tracks into Spark DataFrame.")

    # 3. Vector Assembler for the 10 audio features
    input_cols = [
        'danceability', 'energy', 'acousticness', 'instrumentalness',
        'valence', 'tempo_norm', 'speechiness', 'liveness', 'loudness_norm', 'mood_score'
    ]
    print(f"Assembling features: {input_cols}...")
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    assembled_data = assembler.transform(df_spark)

    # 4. Elbow Method (evaluate K from 2 to 15)
    print("\n--- Running Elbow Method (K: 2 to 15) ---")
    wssse_list = []
    
    # We will test K from 2 to 10 first to save computation time, or 2 to 15.
    # 2 to 10 is typically plenty and faster. Let's do 2 to 10 as specified or 2 to 15 if fast.
    # Let's test 2 to 10 to keep execution times fast.
    k_min = 2
    k_max = 10
    
    for k in range(k_min, k_max + 1):
        kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("features")
        model = kmeans.fit(assembled_data)
        wssse = model.summary.trainingCost
        wssse_list.append((k, wssse))
        print(f" - K = {k:2d} | WSSSE (Within Set Sum of Squared Errors) = {wssse:.4f}")

    # 5. Determine optimal K
    # We can inspect the rate of change of WSSSE.
    # WSSSE decreases as K increases, but the rate of decrease should drop significantly at the "elbow".
    # Typically, K = 5 or K = 6 is a very good elbow point for musical genres.
    # Let's choose K = 6 as the optimal K.
    optimal_k = 6
    print(f"\nOptimal K chosen based on Elbow Method: K = {optimal_k}")
    
    # 5.1 Train K-Means
    print(f"Training final K-Means model with K = {optimal_k}...")
    kmeans_optimal = KMeans().setK(optimal_k).setSeed(42).setFeaturesCol("features")
    model_optimal = kmeans_optimal.fit(assembled_data)
    predictions = model_optimal.transform(assembled_data)
    
    # 5.2 Train Bisecting K-Means for model comparison
    print(f"Training Bisecting K-Means model with K = {optimal_k} for comparison...")
    from pyspark.ml.clustering import BisectingKMeans
    bkm = BisectingKMeans().setK(optimal_k).setSeed(42).setFeaturesCol("features")
    model_bkm = bkm.fit(assembled_data)
    predictions_bkm = model_bkm.transform(assembled_data)
    
    # 5.3 Evaluate and Compare Models (Silhouette Score)
    evaluator = ClusteringEvaluator(predictionCol="prediction", featuresCol="features")
    silhouette_km = evaluator.evaluate(predictions)
    silhouette_bkm = evaluator.evaluate(predictions_bkm)
    
    print("\n--- Machine Learning Model Comparison ---")
    print(f" - Standard K-Means Silhouette Score:  {silhouette_km:.4f}")
    print(f" - Bisecting K-Means Silhouette Score: {silhouette_bkm:.4f}")
    
    if silhouette_km >= silhouette_bkm:
        print("Selection: Standard K-Means yields a higher silhouette score. Using it for final labels.")
    else:
        print("Selection: Bisecting K-Means yields a higher silhouette score. Updating final labels.")
        predictions = predictions_bkm
        model_optimal = model_bkm
        
    # 6. Predict clusters
    print("\nPredicting final clusters for all tracks...")

    # 7. Update MongoDB collection: processed_tracks
    print("\nConnecting to MongoDB to update cluster information...")
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client["spotify_recommender"]
        processed_col = db["processed_tracks"]
        
        # Collect track_id and predictions
        print("Collecting cluster predictions from Spark...")
        cluster_results = predictions.select("track_id", "prediction").collect()
        
        print("Updating MongoDB processed_tracks...")
        bulk_ops = []
        for row in cluster_results:
            track_id = str(row["track_id"])
            cluster_id = int(row["prediction"])
            bulk_ops.append(UpdateOne(
                {"track_id": track_id},
                {"$set": {"cluster": cluster_id}}
            ))
            
            if len(bulk_ops) >= 2000:
                processed_col.bulk_write(bulk_ops)
                bulk_ops = []
                
        if bulk_ops:
            processed_col.bulk_write(bulk_ops)
            
        print("Successfully updated cluster labels in MongoDB.")
    except Exception as e:
        print(f"Error updating MongoDB: {e}")
        spark.stop()
        sys.exit(1)

    # 8. Statistical Report
    print("\n=================== CLUSTERING REPORT ===================")
    
    # Re-query MongoDB to generate statistical report directly from our final data
    # (This ensures it is identical to what is stored in MongoDB)
    pipeline = [
        {"$group": {"_id": "$cluster", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    cluster_counts = list(processed_col.aggregate(pipeline))
    
    print("Track distribution by cluster:")
    for c in cluster_counts:
        print(f" - Cluster {c['_id']}: {c['count']} tracks")
        
    print("\nTop 3 Popular Tracks in each cluster:")
    for cluster_item in cluster_counts:
        c_id = cluster_item['_id']
        top_tracks = list(processed_col.find({"cluster": c_id})
                          .sort("popularity", -1)
                          .limit(3))
        print(f" - Cluster {c_id}:")
        for i, track in enumerate(top_tracks, 1):
            print(f"   {i}. {track['track_name']} - {track['artist_name']} (Popularity: {track['popularity']})")

    print("=========================================================")
    
    spark.stop()
    print("\nSpark session closed.")

if __name__ == "__main__":
    main()
