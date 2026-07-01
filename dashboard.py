import streamlit as st
import networkx as nx
import numpy as np
import folium
from streamlit_folium import st_folium
import copy
from skimage import morphology, io
import sknw
import random
import itertools

# =========================================================
# 0. INITIALIZE UI
# =========================================================
st.set_page_config(page_title="Route Resilience V2.0", layout="wide", page_icon="🛰️")

# =========================================================
# 1. CORE PHYSICS & TOPOLOGY ENGINE
# =========================================================
def heal_islands(graph, distance_threshold, penalty_confidence=0.2):
    components = list(nx.connected_components(graph))
    if len(components) <= 1: return graph
    for comp_a, comp_b in itertools.combinations(components, 2):
        min_dist = float('inf')
        best_pair = None
        for u in comp_a:
            for v in comp_b:
                coord_u, coord_v = graph.nodes[u]['o'], graph.nodes[v]['o']
                dist = np.linalg.norm(coord_u - coord_v)
                if dist < min_dist: min_dist, best_pair = dist, (u, v)
        if best_pair and min_dist < distance_threshold:
            u, v = best_pair
            graph.add_edge(u, v, weight=min_dist, confidence=penalty_confidence, eff_weight=min_dist/penalty_confidence)
    return graph

def merge_close_junctions(graph, merge_threshold):
    undirected = graph.to_undirected()
    for u, v, d in list(undirected.edges(data=True)):
        if d.get('weight', 999) < merge_threshold:
            if graph.has_node(u) and graph.has_node(v):
                graph = nx.contracted_nodes(graph, u, v, self_loops=False)
    return graph

def calculate_network_efficiency(graph):
    n = len(graph)
    if n < 2: return 0.0
    efficiency = 0.0
    paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='eff_weight'))
    for u in graph:
        for v in graph:
            if u != v:
                dist = paths[u].get(v, 0)
                if dist > 0: efficiency += 1.0 / dist
    return efficiency / (n * (n - 1))

def identify_gatekeepers(graph):
    centrality = nx.betweenness_centrality(graph, weight='eff_weight')
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True), centrality

# =========================================================
# 2. DATA PIPELINE (The Trace Extraction Engine)
# =========================================================
# =========================================================
# 2. DATA PIPELINE (The Trace Extraction Engine)
# =========================================================
# =========================================================
# 2. DATA PIPELINE (The Trace Extraction Engine)
# =========================================================
def process_mask_to_graph(image_bytes, bbox, bin_thresh, noise_max, hole_max, heal_dist, merge_dist):
    # Reset the Streamlit file pointer to the beginning so it can be re-read on slider changes
    image_bytes.seek(0)
    
    # 1. Raw Mask Stage (Reverted to your working scikit-image loader)
    raw_img = io.imread(image_bytes, as_gray=True)
    raw_mask = raw_img > bin_thresh
    
    # 2. Healed Mask Stage
    clean_mask = morphology.remove_small_objects(raw_mask, max_size=noise_max)
    healed_mask = morphology.remove_small_holes(clean_mask, max_size=hole_max)
    
    # 3. Skeleton Stage
    skeleton = morphology.skeletonize(healed_mask)
    
    # 4. Vector Graph Stage
    graph = sknw.build_sknw(skeleton)
    for u, v, data in graph.edges(data=True):
        data['confidence'] = 1.0
        data['eff_weight'] = data['weight']
        
    graph = heal_islands(graph, distance_threshold=heal_dist)
    graph = merge_close_junctions(graph, merge_threshold=merge_dist)
    
    # Affine Mapping (Pixels -> Geodetic)
    H, W = healed_mask.shape
    d_lat = (bbox['lat_max'] - bbox['lat_min']) / H
    d_lon = (bbox['lon_max'] - bbox['lon_min']) / W
    for n, data in graph.nodes(data=True):
        if 'o' in data:
            row, col = data['o']
            graph.nodes[n]['pos'] = (bbox['lat_max'] - (row * d_lat), bbox['lon_min'] + (col * d_lon))
            
    # The crucial fix: Convert boolean True/False arrays to 0-255 uint8 images for Streamlit visibility
    disp_raw = (raw_mask * 255).astype(np.uint8)
    disp_heal = (healed_mask * 255).astype(np.uint8)
    disp_skel = (skeleton * 255).astype(np.uint8)
            
    return disp_raw, disp_heal, disp_skel, graph

# =========================================================
# 3. UI LAYOUT
# =========================================================
st.title("🛰️ Route Resilience V2.0: Deep Extraction & Stress Test")

with st.sidebar:
    st.header("1. Calibration & Data")
    bbox = {
        'lat_max': st.number_input("Lat Max", value=13.0980, format="%.4f"),
        'lat_min': st.number_input("Lat Min", value=13.0919, format="%.4f"),
        'lon_min': st.number_input("Lon Min", value=80.1001, format="%.4f"),
        'lon_max': st.number_input("Lon Max", value=80.1101, format="%.4f")
    }
    file = st.file_uploader("Upload Mask", type=['png', 'jpg'])
    
    st.divider()
    st.header("2. Hyperparameter Tuning")
    bin_t = st.slider("Binarization Threshold", 0.0, 1.0, 0.1)
    noise_m = st.slider("Noise Cleaning Area (px)", 10, 1000, 200)
    hole_m = st.slider("Hole Filling Area (px)", 10, 1000, 200)
    heal_d = st.slider("Island Bridge Distance", 5, 200, 50)
    merge_d = st.slider("Junction Merge Radius", 1, 50, 15)
    
    if file:
        if st.button("🚀 Re-Extract Infrastructure", use_container_width=True):
            raw, heal, skel, G = process_mask_to_graph(file, bbox, bin_t, noise_m, hole_m, heal_d, merge_d)
            st.session_state.update({"raw": raw, "heal": heal, "skel": skel, "G_base": G, "G_curr": copy.deepcopy(G), 
                                    "eff_base": calculate_network_efficiency(G), "history": [1.0], "removed": 0})

    st.divider()
    st.header("3. Simulation Controls")
    mode = st.radio("Ablation Strategy", ["Targeted (Gatekeepers)", "Random (Noise)"])
    if st.button("🔥 Execute Ablation Step", type="primary", use_container_width=True):
        if 'G_curr' in st.session_state and len(st.session_state.G_curr) > 1:
            G_sim = st.session_state.G_curr
            target = identify_gatekeepers(G_sim)[0][0][0] if mode == "Targeted (Gatekeepers)" else random.choice(list(G_sim.nodes()))
            G_sim.remove_node(target)
            st.session_state.removed += 1
            st.session_state.history.append(calculate_network_efficiency(G_sim) / st.session_state.eff_base)

# MAIN DASHBOARD
if 'G_curr' in st.session_state:
    st.subheader("🔍 Extraction Trace")
    c1, c2, c3, c4 = st.columns(4)
    c1.image(st.session_state.raw, caption="1. Raw Mask", use_container_width=True)
    c2.image(st.session_state.heal, caption="2. Healed Mask", use_container_width=True)
    c3.image(st.session_state.skel, caption="3. Skeleton", use_container_width=True)
    
    # Logic to show a basic plot of the graph for the 4th stage
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5,5))
    nx.draw(st.session_state.G_base, pos={n: (d['o'][1], -d['o'][0]) for n, d in st.session_state.G_base.nodes(data=True)}, 
            node_size=10, node_color='red', edge_color='green', ax=ax)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    c4.pyplot(fig)

    st.divider()
    m_col, c_col = st.columns([2, 1])
    with m_col:
        st.markdown("### Heatmap")
        gates, c_map = identify_gatekeepers(st.session_state.G_curr)
        
        # 1. Base Map
        m = folium.Map(location=[(bbox['lat_max']+bbox['lat_min'])/2, (bbox['lon_max']+bbox['lon_min'])/2], zoom_start=15, tiles="OpenStreetMap")
        
        # 2. Dark Mode CSS Injection
        dark_mode_css = """
        <style>
        .leaflet-tile {
            filter: invert(100%) hue-rotate(180deg) brightness(95%) contrast(90%);
        }
        </style>
        """
        m.get_root().html.add_child(folium.Element(dark_mode_css))
        
        # 3. Create Transparent RGBA Overlay from the Healed Mask
        # Get the 2D display array (0 for background, 255 for roads)
        heal_img = st.session_state.heal
        H, W = heal_img.shape
        
        # Initialize an empty RGBA array (all zeros = fully transparent black)
        rgba_overlay = np.zeros((H, W, 4), dtype=np.uint8)
        
        # Find where the roads are and set them to White [255, 255, 255] with an Alpha (opacity) of 200
        road_pixels = heal_img > 0
        rgba_overlay[road_pixels] = [255, 255, 255, 200] 
        
        # Define the geospatial boundaries for the image
        img_bounds = [[bbox['lat_min'], bbox['lon_min']], [bbox['lat_max'], bbox['lon_max']]]
        
        # Add the image overlay to the map
        folium.raster_layers.ImageOverlay(
            image=rgba_overlay,
            bounds=img_bounds,
            interactive=False,
            cross_origin=False,
            zindex=1
        ).add_to(m)
                
        # 4. Draw Nodes (Gatekeepers) over the mask
        # 4. Draw Nodes (Gatekeepers) over the mask
        for n, d in st.session_state.G_curr.nodes(data=True):
            if 'pos' in d:
                score = c_map.get(n, 0)
                folium.CircleMarker(
                    d['pos'], 
                    radius=3+(score*30), 
                    color="red" if score*5>1 else "blue", 
                    fill=True,
                    fill_opacity=0.2,  # Restored to the translucent look
                    opacity=0.8        # Keeps the outer border just sharp enough
                ).add_to(m)
                
        st_folium(m, width=700, height=500)
    with c_col:
        st.markdown("### Transition")
        st.line_chart(st.session_state.history, y_label="Resilience Index (R)")
