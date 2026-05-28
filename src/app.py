import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient

# Ensure local import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommender import get_track_features, recommend_songs, FEATURE_COLS

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
        background: linear-gradient(135deg, #09090a 0%, #121214 100%);
        color: #ffffff;
    }
    h1, h2, h3 {
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
    
    /* Sidebar styling */
    .css-1542mo4 {
        background-color: #0c0c0e !important;
    }
    
    /* Custom divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #1DB954, transparent);
        margin: 20px 0;
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

popular_tracks = get_popular_tracks_list()

# Title banner
st.markdown("<h1 style='text-align: center;'>🎵 Spotify <span class='highlight'>Audio-Feature</span> Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #b3b3b3;'>Discover new music based on sound patterns and acoustic features powered by Spark K-Means.</p>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Sidebar layout
st.sidebar.markdown("### 🎛️ Settings")

# 1. Selection Mode
input_mode = st.sidebar.radio("Choose Input Method:", ["Select from Popular Songs", "Paste Spotify URL"])

if input_mode == "Select from Popular Songs":
    if popular_tracks:
        selected_option = st.sidebar.selectbox(
            "Select a Song:",
            popular_tracks,
            format_func=lambda x: x["label"]
        )
        input_url = selected_option["url"]
    else:
        st.sidebar.warning("No tracks found in DB. Please input URL manually.")
        input_url = st.sidebar.text_input("Spotify URL:", "https://open.spotify.com/track/0sf12qNH5qcw8qpgymFOqD")
else:
    input_url = st.sidebar.text_input(
        "Paste Spotify Track URL / ID:",
        "https://open.spotify.com/track/0sf12qNH5qcw8qpgymFOqD"
    )

# 2. Sliders and Filters
st.sidebar.markdown("### 📊 Recommendation Filters")
top_n = st.sidebar.slider("Number of Recommendations:", min_value=5, max_value=20, value=10)

# Release year filter
release_year_range = st.sidebar.slider(
    "Release Year:",
    min_value=1950,
    max_value=2026,
    value=(1980, 2026)
)

# Popularity filter
popularity_range = st.sidebar.slider(
    "Popularity Range:",
    min_value=0,
    max_value=100,
    value=(0, 100)
)

# Submit button
generate = st.sidebar.button("⚡ Generate Recommendations")

# Main content
if generate or 'input_url' in st.session_state or input_url:
    # Save input URL state
    st.session_state['input_url'] = input_url
    
    with st.spinner("Analyzing track audio features..."):
        try:
            # 1. Fetch input track info
            input_track = get_track_features(input_url)
            
            # Display Input Track Information
            st.subheader("🎼 Currently Selected Song")
            
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                # Cover Art
                cover_url = input_track.get('album_cover_url') or "data/covers/cover_synthwave.png"
                st.image(cover_url, width='stretch')
                
            with col2:
                # Metadata
                st.markdown(f"### **{input_track['track_name']}**")
                st.markdown(f"**Artist:** {input_track['artist_name']}")
                st.markdown(f"**Album:** {input_track['album_name']}")
                st.markdown(f"**Release Year:** {input_track['release_year'] or 'Unknown'}")
                st.markdown(f"**Cluster:** <span class='highlight'>Cluster {input_track['cluster']}</span>", unsafe_allow_html=True)
                
                # Popularity slider display
                st.write("**Popularity:**")
                st.progress(int(input_track['popularity']))
                
                # Spotify link button
                st.link_button("🟢 Open in Spotify", input_track['spotify_url'])
                
            with col3:
                # Radar Chart of 10 Audio Features
                st.write("**Acoustic Profile:**")
                
                categories = [c.replace('_norm', '').capitalize() for c in FEATURE_COLS]
                values = [input_track[c] for c in FEATURE_COLS]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    fillcolor='rgba(29, 185, 84, 0.2)',
                    line=dict(color='#1DB954', width=2),
                    name=input_track['track_name']
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            gridcolor='rgba(255,255,255,0.1)'
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)'
                        ),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20),
                    height=260,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, width='stretch')

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            # 2. Get recommendations
            recs = recommend_songs(input_url, top_n=100) # Fetch more to allow year/popularity filtering
            
            # Apply year filter
            recs_filtered = []
            for r in recs:
                # Release year check
                year = r.get('release_year')
                if year is None:
                    # Let it pass if we don't have year
                    year = 2000
                
                # Popularity check
                pop = r.get('popularity', 0)
                
                if (release_year_range[0] <= year <= release_year_range[1]) and \
                   (popularity_range[0] <= pop <= popularity_range[1]):
                    recs_filtered.append(r)
            
            # Truncate to top_n
            recs_filtered = recs_filtered[:top_n]
            
            # Fetch real Spotify album covers in parallel for the final display
            from recommender import fetch_real_covers
            with st.spinner("Fetching real album covers from Spotify..."):
                recs_filtered = fetch_real_covers(recs_filtered)
            
            st.subheader(f"✨ Recommended Songs (Top {len(recs_filtered)})")
            
            if not recs_filtered:
                st.info("No recommendations found matching your criteria. Try adjusting release year or popularity range filters.")
            else:
                for idx, rec in enumerate(recs_filtered, 1):
                    # Outer Card Div
                    rec_cover = rec.get('album_cover_url') or "data/covers/cover_synthwave.png"
                    
                    # Columns inside the card
                    col_img, col_info, col_action = st.columns([1, 5, 2])
                    
                    with col_img:
                        st.image(rec_cover, width=80)
                        
                    with col_info:
                        st.markdown(f"**{idx}. {rec['track_name']}** — {rec['artist_name']}")
                        st.markdown(f"<small>Album: {rec['album_name']} | Match Score: <span class='highlight'>{rec['similarity']:.2%}</span></small>", unsafe_allow_html=True)
                        st.progress(int(rec['popularity']))
                        
                    with col_action:
                        # Link
                        st.link_button("🔗 Listen on Spotify", rec['spotify_url'])
                        
                        # Audio Preview (if exists)
                        preview_url = rec.get('preview_url')
                        if preview_url:
                            st.audio(preview_url, format="audio/mp3")

                    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 10px 0;'>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
