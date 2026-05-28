PHASE 0 — Khởi tạo môi trường
Tôi làm bài tập lớn môn Big Data trên Ubuntu.
Hãy viết script bash: setup_env.sh

Cài đặt toàn bộ:
- Python 3.10+, pip, venv
- MongoDB Community Edition (apt)
- Java 11 (OpenJDK)
- Apache Spark 3.5.x (tar.gz → ~/spark, set SPARK_HOME + PATH vào ~/.bashrc)
- Thư viện Python: pyspark, pymongo, spotipy, pandas, numpy, scikit-learn,
  streamlit, plotly, umap-learn, python-dotenv, joblib, requests

Sau cài xong in: spark-submit --version && mongod --version

PHASE 1 — Thu thập dữ liệu (1.0đ)
Project: Spotify Audio-Feature Recommender bằng Python + MongoDB.
Hãy viết: src/data_collector.py

Yêu cầu:
1. Xác thực Spotify API bằng Client Credentials Flow (Spotipy).
   Đọc SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET từ file .env.

2. Thu thập dữ liệu từ nhiều nguồn để đạt tối thiểu 5.000 bài hát:
   - Lấy toàn bộ track từ 30 playlist nổi tiếng (Top 50 Việt Nam, Global Viral,
     các playlist theo thể loại: Chill, Workout, EDM, Acoustic, V-Pop, K-Pop, Jazz, Hip-Hop).
   - Với mỗi track: gọi sp.audio_features() để lấy 13 đặc trưng âm thanh:
     danceability, energy, key, loudness, mode, speechiness, acousticness,
     instrumentalness, liveness, valence, tempo, duration_ms, time_signature.
   - Lấy thêm metadata: track_name, artist_name, album_name, release_year,
     popularity, preview_url, album_cover_url, spotify_url, track_id.

3. Gọi API theo batch 100 tracks/request (giới hạn của Spotify),
   có retry logic (exponential backoff) khi gặp rate limit 429.

4. Lưu vào MongoDB:
   - Database: spotify_recommender
   - Collection: tracks
   - Tạo unique index trên track_id để tránh duplicate.

5. In báo cáo cuối: tổng tracks thu thập, phân bố theo genre/playlist,
   % tracks có đầy đủ audio features.

Viết kèm .env.example với tất cả key cần thiết.

PHASE 2 — Lưu trữ & Tiền xử lý (1.5đ)
Tiếp theo project Spotify. Hãy viết: src/preprocessing.py

Yêu cầu:
1. Đọc toàn bộ từ MongoDB collection tracks vào Pandas DataFrame.

2. Làm sạch dữ liệu (comment rõ từng bước):
   - Loại bỏ duplicate theo track_id.
   - Xử lý missing values: các feature số null → median của genre tương ứng.
   - Loại bỏ outlier: tempo < 40 hoặc > 220 BPM → thay bằng median.
   - Normalize loudness từ dB (-60 đến 0) về range [0, 1].

3. Feature engineering:
   - Tạo cột energy_valence_ratio = energy / (valence + 0.001).
   - Tạo cột mood_score: kết hợp valence * 0.4 + danceability * 0.3 + energy * 0.3.
   - Tạo cột tempo_category: Slow (<90), Medium (90-130), Fast (>130).
   - Encode cột key và mode thành one-hot.

4. Lưu cleaned data vào:
   - MongoDB collection: processed_tracks
   - File Parquet: data/processed/tracks_clean.parquet

5. In thống kê: correlation matrix của 13 audio features,
   top 10 tracks có popularity cao nhất, phân bố mood_score.

PHASE 3 — Apache Spark + K-Means Clustering (2.5đ)
Tiếp theo project Spotify. Hãy viết: src/spark_clustering.py

Chạy trên PySpark local mode (master="local[*]"):

1. Khởi tạo SparkSession:
   - spark.driver.memory = 2g
   - App name: "Spotify_Clustering"

2. Đọc data/processed/tracks_clean.parquet vào Spark DataFrame.

3. Chuẩn bị features cho clustering:
   - Dùng VectorAssembler với 10 features chính:
     danceability, energy, acousticness, instrumentalness,
     valence, tempo_norm, speechiness, liveness, loudness_norm, mood_score.
   - Dùng StandardScaler để normalize (clustering rất nhạy cảm với scale).

4. Tìm K tối ưu bằng Elbow Method + Silhouette Score:
   - Chạy K-Means với K từ 3 đến 12.
   - Với mỗi K: tính Within Set Sum of Squared Errors (WSSSE) và Silhouette Score.
   - Vẽ biểu đồ Elbow và lưu vào data/elbow_chart.json.
   - Chọn K tối ưu tự động: K có Silhouette Score cao nhất.

5. Train K-Means với K tối ưu:
   - maxIter=50, seed=42.
   - Gán nhãn cluster cho toàn bộ tracks.
   - Phân tích từng cluster: tính mean của mỗi feature → đặt tên cluster
     (ví dụ: Cluster 0 = "Energetic Workout", Cluster 1 = "Chill Acoustic"...)
   - Tên cluster tự động dựa trên rule: nếu energy>0.7 và tempo>120 → "High Energy",
     nếu acousticness>0.6 → "Acoustic & Mellow", v.v.

6. Lưu kết quả:
   - Spark model: models/kmeans_model/
   - DataFrame đã có cluster label → MongoDB collection: clustered_tracks
   - File: data/cluster_analysis.json (thống kê từng cluster)
   - File: data/model_metrics.json (K tối ưu, Silhouette Score, WSSSE)

7. Gọi spark.stop() sau khi xong.

PHASE 4 — Recommendation Engine
Tiếp theo project Spotify. Hãy viết: src/recommender.py

Đây là engine gợi ý khi user paste link Spotify vào:

1. Hàm get_track_features(spotify_url) → dict:
   - Parse track_id từ URL dạng https://open.spotify.com/track/{id}
   - Gọi Spotipy lấy audio_features + metadata của track đó.
   - Áp dụng cùng preprocessing như Phase 2 (load scaler đã fit).

2. Hàm predict_cluster(feature_dict) → cluster_id:
   - Load Spark model từ models/kmeans_model/.
   - Transform feature → dự đoán cluster.
   - Trả về cluster_id + cluster_name + mô tả cluster.

3. Hàm recommend_songs(track_id, cluster_id, n=5) → list:
   - Lọc toàn bộ bài trong cùng cluster từ MongoDB.
   - Loại bỏ chính bài hát đó.
   - Tính cosine similarity giữa audio feature vector của bài input
     với tất cả bài trong cluster.
   - Trả về top N bài similarity cao nhất kèm metadata đầy đủ.

4. Hàm explain_recommendation(input_features, recommended_features) → str:
   - So sánh các feature nổi bật: "Bài này được gợi ý vì có
     Danceability tương đồng (0.82 vs 0.79) và cùng mood Energetic."
   - Dùng rule-based để generate câu giải thích tự nhiên bằng tiếng Việt.

PHASE 5 — Streamlit Dashboard (3.5đ)
Tiếp theo project Spotify. Hãy viết: app/streamlit_app.py

Giao diện dark theme, màu chủ đạo #1DB954 (Spotify green), font Inter.

--- SIDEBAR ---
- Logo Spotify-style + tên project
- Input box: "Dán link bài hát Spotify tại đây"
- Nút "🎵 Phân tích & Gợi ý"
- Thống kê nhanh: tổng số bài trong database, số cluster

--- TAB 1: PHÂN TÍCH BÀI HÁT ---
Sau khi user nhập link:
- Hiện album cover (ảnh từ Spotify API) + tên bài + artist.
- Radar Chart (Plotly) 8 chiều: danceability, energy, valence,
  acousticness, instrumentalness, liveness, speechiness, tempo_norm.
- Gauge chart: mood_score từ 0-100 với nhãn (Buồn / Bình thường / Vui / Sôi động).
- Badge hiện Cluster thuộc về: tên cluster + emoji + mô tả ngắn.

--- TAB 2: GỢI Ý 5 BÀI TƯƠNG TỰ ---
Với mỗi bài được gợi ý, hiện card gồm:
- Album cover thumbnail
- Tên bài + Artist + Năm phát hành
- Similarity score (%) dạng progress bar
- Mini radar chart nhỏ so sánh với bài gốc
- Câu giải thích tại sao được gợi ý (tiếng Việt)
- Nút "▶ Nghe thử" → mở Spotify URL

--- TAB 3: KHÁM PHÁ CLUSTER ---
- Scatter plot 2D dùng UMAP để giảm chiều 10D → 2D,
  mỗi điểm là 1 bài hát, tô màu theo cluster, hover hiện tên bài.
- Bảng thống kê từng cluster: tên, số bài, average features, top 5 bài tiêu biểu.
- Bar chart: so sánh average audio features giữa các cluster.
- Elbow chart + Silhouette Score chart (load từ data/model_metrics.json).

--- TAB 4: EXPLORER ---
- Filter bài hát theo: mood (slider valence), energy (slider), tempo range.
- Kết quả hiện dạng grid cards có thể scroll.
- Nút export playlist sang file CSV.

PHASE 6 — Đóng gói & Demo
Hoàn thiện project Spotify Recommender. Tạo các file:

1. docker-compose.yml:
   - Service mongodb: mongo:7, volume persist
   - Service app: build Dockerfile, port 8501, env_file .env

2. Dockerfile:
   - Base: python:3.10-slim + Java 11
   - COPY requirements.txt → pip install
   - CMD: streamlit run app/streamlit_app.py --server.port 8501

3. run_pipeline.sh — chạy full pipeline có timestamp log:
   python src/data_collector.py
   python src/preprocessing.py
   python src/spark_clustering.py
   streamlit run app/streamlit_app.py

4. README.md chuyên nghiệp:
   - Kiến trúc hệ thống (ASCII diagram: Spotify API → MongoDB → Spark → Streamlit)
   - Giải thích K-Means clustering trong context âm nhạc
   - Bảng kết quả: K tối ưu, Silhouette Score, mô tả từng cluster
   - Hướng dẫn cài đặt + chạy
   - Ảnh demo dashboard (placeholder)

5. requirements.txt với version cụ thể.

6. Cấu trúc thư mục:
   spotify-recommender/
   ├── src/         (data_collector, preprocessing, spark_clustering, recommender)
   ├── app/         (streamlit_app.py)
   ├── models/      (kmeans_model/, scaler.pkl)
   ├── data/        (processed/, elbow_chart.json, cluster_analysis.json)
   ├── notebooks/   (EDA.ipynb)
   └── docs/

PHASE 7 — Báo cáo
Viết báo cáo bài tập lớn môn Big Data: "Khai phá Đặc trưng Âm thanh &
Gợi ý Nhạc Cá nhân hóa với Apache Spark".

Cấu trúc (~15-20 trang, tiếng Việt, học thuật):
1. Giới thiệu: bài toán recommendation system trong âm nhạc,
   tại sao Content-based Filtering tốt hơn Collaborative Filtering cho bài này.
2. Kiến trúc hệ thống & Data Pipeline hoàn chỉnh.
3. Thu thập dữ liệu: Spotify Audio Features API, giải thích ý nghĩa từng feature.
4. Lưu trữ & Tiền xử lý: lý do chọn MongoDB, các bước làm sạch.
5. Apache Spark K-Means: lý thuyết clustering, Elbow Method,
   Silhouette Score, phân tích kết quả từng cluster.
6. Recommendation Engine: Cosine Similarity, giải thích logic gợi ý.
7. Trực quan hóa: mô tả dashboard, UMAP visualization.
8. Kết quả & Đánh giá.
9. Kết luận & Hướng phát triển (thêm Collaborative Filtering, Real-time).

Thứ tự chạy: Phase 0 → 1 → 2 → 3 → 4 (import recommender vào app) → 5 → 6.
Tip cho buổi bảo vệ: Chuẩn bị sẵn 3-4 link Spotify đa dạng thể loại (V-Pop ballad, EDM, acoustic) để demo live — mỗi link cho kết quả cluster khác nhau sẽ rất ấn tượng.