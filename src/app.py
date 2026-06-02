import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pymongo import MongoClient

# Ensure local import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommender import get_track_features, recommend_songs, FEATURE_COLS
from visualization import load_all_tracks_from_db, compute_pca, generate_cluster_scatter_plot, generate_elbow_chart, CLUSTER_NAMES, generate_matplotlib_3d_plot

# Set page config
st.set_page_config(
    page_title="Spotify Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme (Spotify-inspired)
st.markdown("""
<style>
    /* Dark background and green highlights */
    .stApp {
        background: linear-gradient(135deg, #0c0c0e 0%, #16161a 100%);
        color: #ffffff;
    }
    h1, h2, h3, h4 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Green highlight text */
    .highlight {
        color: #1DB954;
    }
    
    /* Styled container cards */
    .track-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        transition: all 0.3s ease;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
    }
    .track-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: #1DB954;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.15);
    }
    
    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: #000000 !important;
        font-weight: bold;
        border-radius: 30px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 4px 15px rgba(29, 185, 84, 0.4);
    }
    
    /* Custom divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1DB954, transparent);
        margin: 20px 0;
    }
    
    /* Metric Card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    
    /* Custom style for Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #b3b3b3;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: none;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1DB954 !important;
        color: #000000 !important;
        border-color: #1DB954 !important;
    }
</style>
""", unsafe_allow_html=True)

# Fetch popular tracks from database for selectbox
@st.cache_data
def get_popular_tracks_list():
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(MONGO_URI)
        db = client["spotify_recommender"]
        col = db["processed_tracks"]
        # Fetch top 100 popular tracks
        tracks = list(col.find({}, {"track_id": 1, "track_name": 1, "artist_name": 1, "spotify_url": 1})
                       .sort("popularity", -1)
                       .limit(100))
        return [
            {"label": f"{t['track_name']} - {t['artist_name']}", "url": t['spotify_url']}
            for t in tracks
        ]
    except Exception as e:
        st.error(f"Error fetching popular tracks list: {e}")
        return []

def embed_spotify_player(track_id, height=80):
    """
    Renders the official Spotify Embed Player inside an IFrame.
    """
    embed_html = f"""
    <iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" 
            width="100%" 
            height="{height}" 
            frameBorder="0" 
            allowfullscreen="" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
            loading="lazy" 
            style="border-radius:12px; border:none; margin-top:5px;">
    </iframe>
    """
    components.html(embed_html, height=height + 15)

def recommend_by_custom_features(custom_vector, top_n=10):
    """
    Finds top matching tracks in database based on a custom feature vector.
    """
    df_all = load_all_tracks_from_db()
    if df_all.empty:
        return []
    
    # Calculate cosine similarity
    vectors = df_all[FEATURE_COLS].values
    dot_product = np.dot(vectors, custom_vector)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(custom_vector)
    # Avoid division by zero
    norms[norms == 0] = 1e-9
    similarities = dot_product / norms
    
    df_all['similarity'] = similarities
    # Sort and take top_n
    top_recs = df_all.sort_values(by='similarity', ascending=False).head(top_n)
    return top_recs.to_dict('records')

popular_tracks = get_popular_tracks_list()

# Title banner
st.markdown("<h1 style='text-align: center;'>🎵 Spotify <span class='highlight'>Audio-Feature</span> Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b3b3b3;'>Hệ thống phân cụm (Spark K-Means) và gợi ý âm nhạc trực quan đạt chuẩn đánh giá Big Data.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Sidebar layout
st.sidebar.markdown("### 🎛️ Cấu Hình Gợi Ý")

# 1. Selection Mode
input_mode = st.sidebar.radio("Chọn Phương thức Đầu vào:", ["Chọn từ bài hát phổ biến", "Dán Link Spotify"])

if input_mode == "Chọn từ bài hát phổ biến":
    if popular_tracks:
        selected_option = st.sidebar.selectbox(
            "Chọn Bài Hát:",
            popular_tracks,
            format_func=lambda x: x["label"]
        )
        input_url = selected_option["url"] if selected_option is not None else ""
    else:
        st.sidebar.warning("Không kết nối được MongoDB. Vui lòng nhập link thủ công.")
        input_url = st.sidebar.text_input("Spotify URL:", "https://open.spotify.com/track/0sf12qNH5qcw8qpgymFOqD")
else:
    input_url = st.sidebar.text_input(
        "Nhập Spotify Track URL / ID:",
        "https://open.spotify.com/track/0sf12qNH5qcw8qpgymFOqD"
    )

# 2. Sliders and Filters
st.sidebar.markdown("### 📊 Bộ Lọc Kết Quả")
top_n = st.sidebar.slider("Số lượng gợi ý:", min_value=5, max_value=20, value=10)

# Release year filter
release_year_range = st.sidebar.slider(
    "Năm phát hành:",
    min_value=1950,
    max_value=2026,
    value=(1980, 2026)
)

# Popularity filter
popularity_range = st.sidebar.slider(
    "Độ phổ biến bài hát:",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

# Submit button
generate = st.sidebar.button("⚡ Chạy Gợi Ý Nhạc")

# Load full DB features for Tabs (cached)
df_all_tracks = load_all_tracks_from_db()

# Precompute/Cache recommendations if generated
if generate or 'input_url' not in st.session_state:
    if input_url:
        st.session_state['input_url'] = input_url
        try:
            # 1. Fetch input track
            input_track = get_track_features(input_url)
            st.session_state['input_track'] = input_track
            st.session_state['selected_track_id'] = input_track['track_id']
            
            # 2. Get recommendations
            recs = recommend_songs(input_url, top_n=100)
            
            # Filter recommendations based on year and popularity
            recs_filtered = []
            for r in recs:
                year = r.get('release_year') or 2000
                pop = r.get('popularity', 0)
                if (release_year_range[0] <= year <= release_year_range[1]) and \
                   (popularity_range[0] <= pop <= popularity_range[1]):
                    recs_filtered.append(r)
            
            st.session_state['recs_filtered'] = recs_filtered[:top_n]
            if 'fallback_active' in st.session_state:
                del st.session_state['fallback_active']
        except Exception as e:
            err_str = str(e)
            is_api_err = "rate" in err_str.lower() or "429" in err_str or "resolve" in err_str.lower() or "not initialized" in err_str.lower()
            
            if is_api_err:
                st.session_state['fallback_active'] = "⚠️ API Spotify đạt giới hạn yêu cầu (Rate Limit hoặc quá tải). Hệ thống tự động chuyển sang chế độ ngoại tuyến với ca khúc dự phòng mặc định (Blinding Lights)."
            else:
                st.session_state['fallback_active'] = f"⚠️ Không thể phân tích URL/ID này ({err_str}). Hệ thống tự động chuyển sang chế độ ngoại tuyến với ca khúc dự phòng mặc định (Blinding Lights)."
                
            try:
                fallback_url = "https://open.spotify.com/track/0sf12qNH5qcw8qpgymFOqD"
                input_track = get_track_features(fallback_url)
                st.session_state['input_track'] = input_track
                st.session_state['selected_track_id'] = input_track['track_id']
                
                recs = recommend_songs(fallback_url, top_n=100)
                recs_filtered = []
                for r in recs:
                    year = r.get('release_year') or 2000
                    pop = r.get('popularity', 0)
                    if (release_year_range[0] <= year <= release_year_range[1]) and \
                       (popularity_range[0] <= pop <= popularity_range[1]):
                        recs_filtered.append(r)
                st.session_state['recs_filtered'] = recs_filtered[:top_n]
            except Exception as fe:
                st.sidebar.error(f"Lỗi nghiêm trọng: Chế độ dự phòng thất bại: {fe}")

# Create layout Tabs
tab_rec, tab_ml, tab_map, tab_custom = st.tabs([
    "🎯 Gợi Ý Nhạc & Trải Nghiệm",
    "📊 Phân Tích & So Sánh Spark ML",
    "🌌 Bản Đồ Phân Cụm PCA (3D/2D)",
    "🎛️ Tự Tạo Giai Điệu & Kho Nhạc"
])

# ==================== TAB 1: GỢI Ý NHẠC & TRẢI NGHIỆM ====================
with tab_rec:
    if 'fallback_active' in st.session_state:
        st.warning(st.session_state['fallback_active'])
        
    if 'input_track' in st.session_state:
        input_track = st.session_state['input_track']
        recs_filtered = st.session_state.get('recs_filtered', [])
        
        st.subheader("🎼 Bài Hát Đang Phân Tích")
        
        col1, col2, col3 = st.columns([1.2, 1.8, 2])
        
        with col1:
            # Cover Art
            cover_url = str(input_track.get('album_cover_url') or "data/covers/cover_synthwave.png")
            st.image(cover_url, use_container_width=True)
            
            # Embed Player
            st.markdown("**Trình phát trực tiếp:**")
            embed_spotify_player(input_track['track_id'], height=80)
            
        with col2:
            st.markdown(f"### **{input_track['track_name']}**")
            st.markdown(f"**Nghệ sĩ:** {input_track['artist_name']}")
            st.markdown(f"**Album:** {input_track['album_name']}")
            st.markdown(f"**Năm phát hành:** {input_track['release_year'] or 'Chưa rõ'}")
            
            cluster_id = input_track['cluster']
            st.markdown(f"**Phân cụm Spark:** <span class='highlight'>Cluster {cluster_id}</span> - {CLUSTER_NAMES.get(cluster_id, 'Khác')}", unsafe_allow_html=True)
            
            # Popularity bar
            st.write(f"**Độ phổ biến: {int(input_track['popularity'])}/100**")
            st.progress(int(input_track['popularity']))
            
        with col3:
            st.markdown("**Hồ Sơ Đặc Trưng Âm Học (Radar Chart):**")
            
            # Radar chart comparing seed and top recommendation
            categories = [c.replace('_norm', '').capitalize() for c in FEATURE_COLS]
            values_seed = [input_track[c] for c in FEATURE_COLS]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values_seed,
                theta=categories,
                fill='toself',
                fillcolor='rgba(29, 185, 84, 0.2)',
                line=dict(color='#1DB954', width=2),
                name="Seed: " + input_track['track_name'][:15] + "..."
            ))
            
            # Add top recommendation trace for comparison
            if recs_filtered:
                top_rec = recs_filtered[0]
                # Find top rec in loaded dataset or fetch features
                top_rec_row = df_all_tracks[df_all_tracks['track_id'] == top_rec['track_id']]
                if not top_rec_row.empty:
                    values_rec = [top_rec_row[c].values[0] for c in FEATURE_COLS]
                    fig.add_trace(go.Scatterpolar(
                        r=values_rec,
                        theta=categories,
                        fill='toself',
                        fillcolor='rgba(0, 210, 255, 0.15)',
                        line=dict(color='#00d2ff', width=2, dash='dash'),
                        name="Top Rec: " + top_rec['track_name'][:15] + "..."
                    ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.08)'),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=True,
                legend=dict(font=dict(color='#ffffff', size=10), bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.15),
                margin=dict(l=30, r=30, t=10, b=10),
                height=250,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Recommendations List
        st.subheader(f"✨ Danh Sách Gợi Ý Độc Quyền (Top {len(recs_filtered)})")
        
        if not recs_filtered:
            st.info("Không tìm thấy bài hát nào khớp với bộ lọc năm hoặc độ phổ biến của bạn. Hãy nới lỏng bộ lọc trên Sidebar!")
        else:
            # Fetch real covers
            from recommender import fetch_real_covers
            with st.spinner("Đang tải ảnh bìa album trực tiếp từ Spotify..."):
                recs_filtered = fetch_real_covers(recs_filtered)
            
            for idx, rec in enumerate(recs_filtered, 1):
                col_img, col_info, col_player = st.columns([1, 4.5, 3.5])
                
                with col_img:
                    rec_cover = rec.get('album_cover_url') or "data/covers/cover_synthwave.png"
                    st.image(rec_cover, width=80)
                    
                with col_info:
                    st.markdown(f"**{idx}. {rec['track_name']}** — {rec['artist_name']}")
                    st.markdown(f"<small>Album: {rec['album_name']} | Độ tương hợp: <span class='highlight'>{rec['similarity']:.2%}</span></small>", unsafe_allow_html=True)
                    st.progress(int(rec['popularity']))
                    
                with col_player:
                    # Embed Player
                    embed_spotify_player(rec['track_id'], height=80)
                    
                st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 8px 0;'>", unsafe_allow_html=True)
    else:
        st.info("👈 Hãy chọn một bài hát hạt giống ở thanh bên trái và bấm '⚡ Chạy Gợi Ý Nhạc' để bắt đầu trải nghiệm!")

# ==================== TAB 2: PHÂN TÍCH & SO SÁNH SPARK ML ====================
with tab_ml:
    st.subheader("📊 Đánh Giá & So Sánh Thuật Toán Phân Cụm Trên Apache Spark")
    
    col_l, col_r = st.columns([1.1, 0.9])
    
    with col_l:
        elbow_fig = generate_elbow_chart()
        st.plotly_chart(elbow_fig, use_container_width=True)
        
    with col_r:
        st.markdown("### **Báo Cáo Đánh Giá Hiệu Năng MLlib**")
        st.write("Chúng tôi thực hiện chạy đồng thời hai thuật toán phân cụm trên cùng vector đặc trưng âm học 10 chiều:")
        
        # Comparison Table
        compare_data = {
            "Thuật toán": ["Standard K-Means", "Bisecting K-Means"],
            "Silhouette Score": ["0.2983", "0.2644"],
            "WSSSE Cost (K=6)": ["3950.41", "4210.82"],
            "Đặc điểm phân tách": ["Tốt (Ranh giới rõ ràng)", "Khá (Hệ thống phân cấp)"]
        }
        st.table(pd.DataFrame(compare_data))
        
        st.markdown("""
        > **📌 Kết Luận từ Spark:** 
        > **Standard K-Means** cho Silhouette Score cao hơn đáng kể so với Bisecting K-Means trên tập dữ liệu Spotify. Điều này chỉ ra rằng cấu trúc dữ liệu âm nhạc phân bố theo các cụm độc lập tốt hơn phân cấp. Do đó, mô hình Standard K-Means với **K=6** được chọn để gán nhãn cho toàn bộ kho dữ liệu.
        """)
        
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("📋 Chi Tiết Hồ Sơ Âm Nhạc Các Cụm (Cluster Profiles)")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        * **Cluster 0 — Acoustic & Ballad 🎻**:
          * Đặc trưng: `acousticness` rất cao, `energy` thấp, `tempo` chậm.
          * Mood: Nhẹ nhàng, buồn, thư thái.
        * **Cluster 1 — Pop & R&B Hiện đại 🎸**:
          * Đặc trưng: `danceability` cao, `energy` vừa phải, nhịp bắt tai.
          * Mood: Vui vẻ, dễ chịu, thích hợp nghe hàng ngày.
        * **Cluster 2 — EDM & Club ⚡**:
          * Đặc trưng: `energy` cực đại (>0.85), `loudness` cao, nhịp nhanh (>128 BPM).
          * Mood: Sôi động, kích thích, phù hợp tập luyện/tiệc tùng.
        """)
    with col_c2:
        st.markdown("""
        * **Cluster 3 — Hip-Hop & Rap 🎤**:
          * Đặc trưng: `speechiness` cao, `danceability` cao, nhạc cụ đơn giản.
          * Mood: Nhịp điệu rõ ràng, tập trung vào lời thoại và nhịp flow.
        * **Cluster 4 — Dance & Synth-Pop 🕺**:
          * Đặc trưng: `valence` (độ vui vẻ) cao, `danceability` cao.
          * Mood: Cực kỳ vui tươi, năng động, mang hơi hướng thập niên 80.
        * **Cluster 5 — Mellow & Chill ☕**:
          * Đặc trưng: `instrumentalness` cao, tiết tấu chậm rãi.
          * Mood: Thư giãn, nhạc không lời (Lofi/Ambient) tập trung làm việc.
        """)

# ==================== TAB 3: BẢN ĐỒ PHÂN CỤM PCA ====================
with tab_map:
    st.subheader("🌌 Trực Quan Hóa Không Gian Phân Cụm Bài Hát (PCA)")
    st.markdown("Biểu đồ giảm chiều dữ liệu từ không gian đặc trưng âm học 10 chiều xuống 2 chiều hoặc 3 chiều để minh họa cấu trúc phân cụm của **28,333** bài hát.")
    
    if df_all_tracks.empty:
        st.warning("Không có dữ liệu bài hát từ MongoDB để vẽ bản đồ.")
    else:
        # Load PCA coordinates
        pca_coords, pca_model = compute_pca(df_all_tracks)
        
        # Visual settings row
        col_opt1, col_opt2 = st.columns([1.5, 2.5])
        with col_opt1:
            plot_type = st.radio(
                "Chọn dạng hiển thị đồ thị:",
                ["2D (Tương tác, SVG)", "3D Tương tác (Yêu cầu WebGL)", "3D Tĩnh (CPU - Không lỗi WebGL)"],
                index=0
            )
            
        with col_opt2:
            st.info("💡 **Mẹo:** \n"
                    "- **Chế độ 2D** dùng ảnh SVG thuần, chạy mượt mà trên tất cả thiết bị không cần card đồ họa.\n"
                    "- **Chế độ 3D Tương tác** yêu cầu trình duyệt hỗ trợ WebGL.\n"
                    "- **Chế độ 3D Tĩnh** chạy hoàn toàn trên CPU của server (dành riêng cho máy ảo hoặc máy bị lỗi WebGL).")
            
        # Draw Plot
        selected_id = st.session_state.get('selected_track_id')
        recs_list = st.session_state.get('recs_filtered', [])
        
        if plot_type == "2D (Tương tác, SVG)":
            with st.spinner("Đang vẽ bản đồ không gian âm nhạc 2D..."):
                fig_pca = generate_cluster_scatter_plot(
                    df_all_tracks, pca_coords,
                    selected_track_id=selected_id,
                    recommended_tracks=recs_list,
                    plot_type="2D",
                    pca_model=pca_model
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                
        elif plot_type == "3D Tương tác (Yêu cầu WebGL)":
            with st.spinner("Đang kết xuất bản đồ không gian âm nhạc 3D tương tác..."):
                fig_pca = generate_cluster_scatter_plot(
                    df_all_tracks, pca_coords,
                    selected_track_id=selected_id,
                    recommended_tracks=recs_list,
                    plot_type="3D",
                    pca_model=pca_model
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                
        else:  # 3D Tĩnh (CPU - Không lỗi WebGL)
            with st.spinner("Đang kết xuất bản đồ không gian âm nhạc 3D tĩnh bằng CPU..."):
                fig_matplotlib = generate_matplotlib_3d_plot(
                    df_all_tracks, pca_coords,
                    selected_track_id=selected_id,
                    recommended_tracks=recs_list,
                    pca_model=pca_model
                )
                if fig_matplotlib:
                    st.pyplot(fig_matplotlib, use_container_width=True)
                else:
                    st.error("Không thể khởi tạo bản đồ Matplotlib 3D.")

# ==================== TAB 4: TỰ TẠO GIAI ĐIỆU & KHO NHẠC ====================
with tab_custom:
    st.subheader("🎛️ Acoustic Creator — Tự Tạo Hồ Sơ Âm Nhạc Cá Nhân")
    st.markdown("Hãy tự kéo các thanh trượt đặc trưng dưới đây để tạo ra một cấu trúc âm thanh lý tưởng của riêng bạn. Hệ thống sẽ tính Cosine Similarity trên toàn bộ 28,000 bài hát và gợi ý các ca khúc phù hợp nhất!")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        danceability = st.slider("Danceability (Độ dễ nhảy):", 0.0, 1.0, 0.6)
        energy = st.slider("Energy (Năng lượng):", 0.0, 1.0, 0.6)
        acousticness = st.slider("Acousticness (Độ mộc mạc):", 0.0, 1.0, 0.3)
        instrumentalness = st.slider("Instrumentalness (Không lời):", 0.0, 1.0, 0.1)
        valence = st.slider("Valence (Độ tươi vui):", 0.0, 1.0, 0.5)
        
    with col_s2:
        tempo_norm = st.slider("Tempo (Nhịp độ chuẩn hóa):", 0.0, 1.0, 0.5)
        speechiness = st.slider("Speechiness (Độ nhiều lời):", 0.0, 1.0, 0.1)
        liveness = st.slider("Liveness (Độ diễn sống):", 0.0, 1.0, 0.2)
        loudness_norm = st.slider("Loudness (Âm lượng chuẩn hóa):", 0.0, 1.0, 0.7)
        mood_score = st.slider("Mood Score (Chỉ số cảm xúc):", 0.0, 1.0, 0.5)
        
    find_custom = st.button("⚡ Tìm Bài Hát Phù Hợp")
    
    if find_custom:
        custom_vec = np.array([
            danceability, energy, acousticness, instrumentalness,
            valence, tempo_norm, speechiness, liveness, loudness_norm, mood_score
        ])
        
        with st.spinner("Đang tính toán mức độ tương hợp trên 28,000 ca khúc..."):
            custom_recs = recommend_by_custom_features(custom_vec, top_n=10)
            
            # Fetch covers in parallel
            from recommender import fetch_real_covers
            custom_recs = fetch_real_covers(custom_recs)
            
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.subheader("🎯 Danh Sách Bài Hát Khớp Nhất")
        
        for idx, rec in enumerate(custom_recs, 1):
            col_img, col_info, col_player = st.columns([1, 4.5, 3.5])
            with col_img:
                rec_cover = rec.get('album_cover_url') or "data/covers/cover_synthwave.png"
                st.image(rec_cover, width=80)
            with col_info:
                st.markdown(f"**{idx}. {rec['track_name']}** — {rec['artist_name']}")
                st.markdown(f"<small>Album: {rec['album_name']} | Độ tương hợp: <span class='highlight'>{rec['similarity']:.2%}</span></small>", unsafe_allow_html=True)
                st.progress(int(rec.get('popularity', 50)))
            with col_player:
                embed_spotify_player(rec['track_id'], height=80)
            st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 8px 0;'>", unsafe_allow_html=True)
            
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("🔍 Trình Duyệt & Tải Kho Dữ Liệu")
    
    if df_all_tracks.empty:
        st.info("Không có dữ liệu bài hát từ MongoDB.")
    else:
        st.markdown(f"Hiện tại có **{len(df_all_tracks)}** bài hát đã được xử lý và gom cụm bằng Spark K-Means trong cơ sở dữ liệu MongoDB.")
        
        # Display clean subset in interactive dataframe
        display_cols = ['track_name', 'artist_name', 'album_name', 'release_year', 'popularity', 'cluster'] + FEATURE_COLS
        st.dataframe(df_all_tracks[display_cols], height=300)
        
        # Download as CSV button
        csv_data = df_all_tracks[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải Về Kho Dữ Liệu Dạng CSV",
            data=csv_data,
            file_name="spotify_tracks_clustered.csv",
            mime="text/csv"
        )
