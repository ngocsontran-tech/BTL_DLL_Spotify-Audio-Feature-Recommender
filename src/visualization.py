import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from pymongo import MongoClient
import streamlit as st

FEATURE_COLS = [
    'danceability', 'energy', 'acousticness', 'instrumentalness',
    'valence', 'tempo_norm', 'speechiness', 'liveness', 'loudness_norm', 'mood_score'
]

CLUSTER_NAMES = {
    0: "Acoustic & Ballad (Chậm/Sâu lắng)",
    1: "Pop & R&B Hiện đại (Vừa phải/Bắt tai)",
    2: "EDM & Club (Năng lượng cực cao)",
    3: "Hip-Hop & Rap (Nhiều lời/Nhịp rõ)",
    4: "Dance & Synth-Pop (Sôi động/Vui vẻ)",
    5: "Mellow & Chill (Nhẹ nhàng/Thư giãn)"
}

CLUSTER_COLORS = {
    0: "#1DB954",  # Spotify Green
    1: "#00d2ff",  # Neon Blue
    2: "#ff007f",  # Neon Pink
    3: "#ffaa00",  # Neon Orange
    4: "#b026ff",  # Purple
    5: "#98ff98"   # Mint Green
}

@st.cache_data
def load_all_tracks_from_db():
    """
    Loads all tracks with their features and cluster labels from MongoDB.
    Cached to prevent repeating database queries.
    """
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    try:
        client = MongoClient(MONGO_URI)
        db = client["spotify_recommender"]
        col = db["processed_tracks"]
        
        # Load necessary columns to save memory
        fields = {
            "track_id": 1, "track_name": 1, "artist_name": 1, "album_name": 1,
            "popularity": 1, "cluster": 1, "release_year": 1
        }
        for col_name in FEATURE_COLS:
            fields[col_name] = 1
            
        tracks = list(col.find({}, fields))
        df = pd.DataFrame(tracks)
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])
        return df
    except Exception as e:
        print(f"Error loading tracks from MongoDB in visualization: {e}")
        return pd.DataFrame()

@st.cache_resource
def compute_pca(df):
    """
    Fits PCA on the track features and returns the coordinates.
    Cached to prevent recalculating PCA on every run.
    """
    if df.empty:
        return None, None
    
    # Extract features
    x = df[FEATURE_COLS].values
    
    # Fit PCA (3 components to support both 2D and 3D)
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(x)
    
    return pca_coords, pca

def generate_cluster_scatter_plot(df, pca_coords, selected_track_id=None, recommended_tracks=None, plot_type="2D", pca_model=None):
    """
    Generates a Plotly scatter plot (2D or 3D) of the song cluster space.
    Highlights the selected song and its recommendations, drawing connecting lines.
    """
    if df.empty or pca_coords is None:
        return go.Figure()
    
    # Prepare data for plotting
    plot_df = df.copy()
    plot_df['PCA1'] = pca_coords[:, 0]
    plot_df['PCA2'] = pca_coords[:, 1]
    plot_df['PCA3'] = pca_coords[:, 2]
    
    # Map cluster names and colors
    plot_df['cluster_name'] = plot_df['cluster'].map(lambda x: CLUSTER_NAMES.get(x, f"Cluster {x}"))
    
    # If the selected track is not in the database (e.g. pasted new Spotify link),
    # project it dynamically using the fitted PCA model.
    if selected_track_id and selected_track_id not in plot_df['track_id'].values:
        input_track = st.session_state.get('input_track')
        if input_track and input_track.get('track_id') == selected_track_id and pca_model is not None:
            try:
                x_new = np.array([[input_track[col] for col in FEATURE_COLS]])
                coords_new = pca_model.transform(x_new)[0]
                new_row = pd.DataFrame([{
                    'track_id': selected_track_id,
                    'track_name': input_track['track_name'],
                    'artist_name': input_track['artist_name'],
                    'album_name': input_track['album_name'],
                    'popularity': input_track['popularity'],
                    'cluster': input_track['cluster'],
                    'PCA1': coords_new[0],
                    'PCA2': coords_new[1],
                    'PCA3': coords_new[2],
                    'cluster_name': CLUSTER_NAMES.get(input_track['cluster'], f"Cluster {input_track['cluster']}")
                }])
                plot_df = pd.concat([plot_df, new_row], ignore_index=True)
            except Exception as e:
                print(f"Error projecting selected track in PCA scatter plot: {e}")
                
    is_3d = (plot_type == "3D")
    
    # If using 2D (SVG go.Scatter), sample background to 5,000 points to prevent browser freeze
    if not is_3d and len(plot_df) > 5000:
        special_ids = []
        if selected_track_id:
            special_ids.append(selected_track_id)
        if recommended_tracks:
            special_ids.extend([r['track_id'] for r in recommended_tracks])
            
        special_df = plot_df[plot_df['track_id'].isin(special_ids)]
        bg_df = plot_df[~plot_df['track_id'].isin(special_ids)]
        
        # Sample 5000 from background
        bg_sampled = bg_df.sample(n=min(5000, len(bg_df)), random_state=42)
        plot_df = pd.concat([bg_sampled, special_df])
        
        # Recalculate index
        plot_df = plot_df.reset_index(drop=True)
    
    fig = go.Figure()
    
    # Draw background cluster points
    for cluster_id in sorted(plot_df['cluster'].unique()):
        cluster_df = plot_df[plot_df['cluster'] == cluster_id]
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        color = CLUSTER_COLORS.get(cluster_id, "#888888")
        
        hover_text = (
            "<b>" + cluster_df['track_name'] + "</b><br>" +
            "Artist: " + cluster_df['artist_name'] + "<br>" +
            "Album: " + cluster_df['album_name'] + "<br>" +
            "Popularity: " + cluster_df['popularity'].astype(str)
        )
        
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=cluster_df['PCA1'],
                y=cluster_df['PCA2'],
                z=cluster_df['PCA3'],
                mode='markers',
                marker=dict(size=3, color=color, opacity=0.4),
                name=cluster_name,
                text=hover_text,
                hoverinfo='text'
            ))
        else:
            # SVG-based go.Scatter (no WebGL required)
            fig.add_trace(go.Scatter(
                x=cluster_df['PCA1'],
                y=cluster_df['PCA2'],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.4),
                name=cluster_name,
                text=hover_text,
                hoverinfo='text'
            ))
            
    # Highlight selected song and recommended songs
    if selected_track_id:
        selected_row = plot_df[plot_df['track_id'] == selected_track_id]
        if not selected_row.empty:
            s_x = selected_row['PCA1'].values[0]
            s_y = selected_row['PCA2'].values[0]
            s_z = selected_row['PCA3'].values[0]
            
            s_text = f"🎯 <b>SEED: {selected_row['track_name'].values[0]}</b><br>Artist: {selected_row['artist_name'].values[0]}"
            
            # Highlight recommended tracks
            rec_coords = []
            if recommended_tracks:
                for idx, rec in enumerate(recommended_tracks, 1):
                    rec_row = plot_df[plot_df['track_id'] == rec['track_id']]
                    if not rec_row.empty:
                        r_x = rec_row['PCA1'].values[0]
                        r_y = rec_row['PCA2'].values[0]
                        r_z = rec_row['PCA3'].values[0]
                        rec_coords.append((r_x, r_y, r_z, rec))
                        
                        # Draw line from seed to recommendation
                        if is_3d:
                            fig.add_trace(go.Scatter3d(
                                x=[s_x, r_x], y=[s_y, r_y], z=[s_z, r_z],
                                mode='lines',
                                line=dict(color='#ffffff', width=2, dash='dash'),
                                showlegend=False,
                                hoverinfo='none'
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=[s_x, r_x], y=[s_y, r_y],
                                mode='lines',
                                line=dict(color='#ffffff', width=1.5, dash='dash'),
                                showlegend=False,
                                hoverinfo='none'
                            ))
            
            # Draw recommended points as a single trace to keep the legend clean
            if rec_coords:
                rx_list = [c[0] for c in rec_coords]
                ry_list = [c[1] for c in rec_coords]
                rz_list = [c[2] for c in rec_coords]
                r_texts = [f"✨ <b>REC: {c[3]['track_name']}</b><br>Artist: {c[3]['artist_name']}<br>Similarity: {c[3]['similarity']:.2%}" for c in rec_coords]
                
                if is_3d:
                    fig.add_trace(go.Scatter3d(
                        x=rx_list, y=ry_list, z=rz_list,
                        mode='markers',
                        marker=dict(size=8, color='#00d2ff', symbol='circle', line=dict(color='#ffffff', width=1)),
                        name="Bài hát gợi ý (Recommendations)",
                        text=r_texts,
                        hoverinfo='text'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=rx_list, y=ry_list,
                        mode='markers',
                        marker=dict(size=10, color='#00d2ff', symbol='circle', line=dict(color='#ffffff', width=1.5)),
                        name="Bài hát gợi ý (Recommendations)",
                        text=r_texts,
                        hoverinfo='text'
                    ))
            
            # Draw selected point last so it's on top
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=[s_x], y=[s_y], z=[s_z],
                    mode='markers',
                    marker=dict(size=12, color='#ffffff', symbol='diamond', line=dict(color='#1DB954', width=2)),
                    name="Bài hát đang chọn (Seed)",
                    text=s_text,
                    hoverinfo='text'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[s_x], y=[s_y],
                    mode='markers',
                    marker=dict(size=14, color='#ffffff', symbol='diamond', line=dict(color='#1DB954', width=2)),
                    name="Bài hát đang chọn (Seed)",
                    text=s_text,
                    hoverinfo='text'
                ))
                
    # Update layout aesthetics
    fig.update_layout(
        paper_bgcolor='rgba(15,15,17,0.85)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='rgba(0,0,0,0)',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if is_3d:
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='PCA 1', gridcolor='rgba(255,255,255,0.05)', backgroundcolor='rgba(0,0,0,0)', color='#ffffff'),
                yaxis=dict(title='PCA 2', gridcolor='rgba(255,255,255,0.05)', backgroundcolor='rgba(0,0,0,0)', color='#ffffff'),
                zaxis=dict(title='PCA 3', gridcolor='rgba(255,255,255,0.05)', backgroundcolor='rgba(0,0,0,0)', color='#ffffff'),
            )
        )
    else:
        fig.update_xaxes(title='PCA Component 1', color='#ffffff', gridcolor='rgba(255,255,255,0.05)', zeroline=False)
        fig.update_yaxes(title='PCA Component 2', color='#ffffff', gridcolor='rgba(255,255,255,0.05)', zeroline=False)
        
    return fig

def generate_elbow_chart():
    """
    Generates a static/dynamic Plotly chart for the Elbow Method
    evaluating standard K-Means cluster cost (WSSSE).
    """
    # WSSSE values from the actual spark clustering run log:
    # K=2: 6680.87, K=3: 5693.30, K=4: 4945.71, K=5: 4298.80, K=6: 3950.41,
    # K=7: 3680.12, K=8: 3450.98, K=9: 3270.43, K=10: 3120.98
    k_values = list(range(2, 11))
    wssse_values = [6680.87, 5693.30, 4945.71, 4298.80, 3950.41, 3680.12, 3450.98, 3270.43, 3120.98]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_values,
        y=wssse_values,
        mode='lines+markers',
        line=dict(color='#1DB954', width=3),
        marker=dict(size=8, color='#ffffff', line=dict(color='#1DB954', width=2)),
        name='WSSSE'
    ))
    
    # Highlight Elbow Point (K=6)
    fig.add_trace(go.Scatter(
        x=[6],
        y=[3950.41],
        mode='markers',
        marker=dict(size=14, color='#ff007f', symbol='circle', line=dict(color='#ffffff', width=2)),
        name='Optimal K=6 (Elbow Point)'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Elbow Method for Optimal K selection</b>',
            font=dict(color='#ffffff', size=16)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        legend=dict(font=dict(color='#ffffff'), bgcolor='rgba(0,0,0,0)'),
        height=300
    )
    
    fig.update_xaxes(title='Number of Clusters (K)', color='#ffffff', gridcolor='rgba(255,255,255,0.05)')
    fig.update_yaxes(title='WSSSE (Cost)', color='#ffffff', gridcolor='rgba(255,255,255,0.05)')
    
    return fig

def generate_matplotlib_3d_plot(df, pca_coords, selected_track_id=None, recommended_tracks=None, pca_model=None):
    """
    Generates a static 3D scatter plot of the song cluster space using Matplotlib (CPU rendering).
    No WebGL required.
    """
    import matplotlib.pyplot as plt
    
    if df.empty or pca_coords is None:
        return None
        
    plot_df = df.copy()
    plot_df['PCA1'] = pca_coords[:, 0]
    plot_df['PCA2'] = pca_coords[:, 1]
    plot_df['PCA3'] = pca_coords[:, 2]
    
    # If the selected track is not in the database (e.g. pasted new Spotify link),
    # project it dynamically using the fitted PCA model.
    if selected_track_id and selected_track_id not in plot_df['track_id'].values:
        input_track = st.session_state.get('input_track')
        if input_track and input_track.get('track_id') == selected_track_id and pca_model is not None:
            try:
                x_new = np.array([[input_track[col] for col in FEATURE_COLS]])
                coords_new = pca_model.transform(x_new)[0]
                new_row = pd.DataFrame([{
                    'track_id': selected_track_id,
                    'track_name': input_track['track_name'],
                    'artist_name': input_track['artist_name'],
                    'album_name': input_track['album_name'],
                    'popularity': input_track['popularity'],
                    'cluster': input_track['cluster'],
                    'PCA1': coords_new[0],
                    'PCA2': coords_new[1],
                    'PCA3': coords_new[2]
                }])
                plot_df = pd.concat([plot_df, new_row], ignore_index=True)
            except Exception as e:
                print(f"Error projecting selected track in Matplotlib 3D: {e}")
    
    # Sample background to 3,000 points to keep Matplotlib rendering fast
    if len(plot_df) > 3000:
        special_ids = []
        if selected_track_id:
            special_ids.append(selected_track_id)
        if recommended_tracks:
            special_ids.extend([r['track_id'] for r in recommended_tracks])
            
        special_df = plot_df[plot_df['track_id'].isin(special_ids)]
        bg_df = plot_df[~plot_df['track_id'].isin(special_ids)]
        bg_sampled = bg_df.sample(n=min(3000, len(bg_df)), random_state=42)
        plot_df = pd.concat([bg_sampled, special_df]).reset_index(drop=True)
        
    # Setup dark style matplotlib figure
    fig = plt.figure(figsize=(10, 7), facecolor='#0c0c0e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0c0c0e')
    
    # Clean axes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1f1f23')
    ax.yaxis.pane.set_edgecolor('#1f1f23')
    ax.zaxis.pane.set_edgecolor('#1f1f23')
    
    # Grid lines
    ax.grid(True, color='rgba(255, 255, 255, 0.05)', linestyle=':')
    
    # Tick colors
    ax.tick_params(colors='#ffffff', labelsize=8)
    
    # Axis labels
    ax.set_xlabel('PCA 1', color='#ffffff', fontsize=10, labelpad=8)
    ax.set_ylabel('PCA 2', color='#ffffff', fontsize=10, labelpad=8)
    ax.set_zlabel('PCA 3', color='#ffffff', fontsize=10, labelpad=8)
    
    # Plot background clusters
    for cluster_id in sorted(plot_df['cluster'].unique()):
        cluster_df = plot_df[plot_df['cluster'] == cluster_id]
        color = CLUSTER_COLORS.get(cluster_id, "#888888")
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        
        ax.scatter(
            cluster_df['PCA1'],
            cluster_df['PCA2'],
            cluster_df['PCA3'],
            c=color,
            s=12,
            alpha=0.4,
            label=cluster_name,
            edgecolors='none'
        )
        
    # Highlight selected song and recommended songs
    if selected_track_id:
        selected_row = plot_df[plot_df['track_id'] == selected_track_id]
        if not selected_row.empty:
            s_x = selected_row['PCA1'].values[0]
            s_y = selected_row['PCA2'].values[0]
            s_z = selected_row['PCA3'].values[0]
            
            # Highlight recommended tracks
            if recommended_tracks:
                for rec in recommended_tracks:
                    rec_row = plot_df[plot_df['track_id'] == rec['track_id']]
                    if not rec_row.empty:
                        r_x = rec_row['PCA1'].values[0]
                        r_y = rec_row['PCA2'].values[0]
                        r_z = rec_row['PCA3'].values[0]
                        
                        # Draw line from seed to recommendation
                        ax.plot([s_x, r_x], [s_y, r_y], [s_z, r_z], color='#ffffff', linestyle='--', linewidth=1.5, alpha=0.8)
                        # Plot recommendation point
                        ax.scatter([r_x], [r_y], [r_z], color='#00d2ff', marker='o', s=80, edgecolors='#ffffff', linewidths=1.5, zorder=5)
            
            # Plot seed point last
            ax.scatter([s_x], [s_y], [s_z], color='#ffffff', marker='D', s=120, edgecolors='#1DB954', linewidths=2.0, zorder=10, label="Selected Song (Seed)")
            
    # Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), facecolor='#16161a', edgecolor='#1f1f23')
    plt.setp(legend.get_texts(), color='#ffffff', fontsize=8)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig
