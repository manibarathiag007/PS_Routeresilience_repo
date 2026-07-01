import streamlit as st
import networkx as nx
import numpy as np
import folium
from streamlit_folium import st_folium
from skimage import morphology, io
import sknw
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

st.set_page_config(page_title="Route Resilience Pipeline", layout="wide", page_icon="🛰️")

# --- CORE PHYSICS ENGINE ---
def calculate_network_efficiency(graph, weight_attr='eff_weight'):
    n = len(graph)
    if n < 2: return 0.0
    efficiency = 0.0
    # Using all-pairs-dijkstra to find shortest path lengths
    paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight_attr))
    for u in graph:
        for v in graph:
            if u != v:
                try:
                    dist = paths[u][v]
                    if dist > 0: efficiency += 1.0 / dist
                except KeyError: pass
    return efficiency / (n * (n - 1))

def identify_gatekeepers(graph):
    return nx.betweenness_centrality(graph, weight='eff_weight')

def mask_to_png_base64(binary_mask, centrality_map, graph):
    """Maps node centrality to the road pixels and converts to a base64 PNG."""
    h, w = binary_mask.shape
    cmap = plt.get_cmap('coolwarm')
    
    node_coords = np.array([graph.nodes[n]['o'] for n in graph.nodes])
    grid_y, grid_x = np.indices((h, w))
    vals = np.array([centrality_map.get(n, 0) for n in graph.nodes])
    
    # Assign each pixel to the nearest node
    dist_sq = (grid_x[:, :, np.newaxis] - node_coords[:, 1])**2 + (grid_y[:, :, np.newaxis] - node_coords[:, 0])**2
    nearest_node_idx = np.argmin(dist_sq, axis=2)
    pixel_centrality = vals[nearest_node_idx]
    
    # Apply colormap
    colors = cmap(pixel_centrality * 5)
    colors[:, :, 3] = binary_mask * 0.6  # Apply transparency to road pixels
    
    # Convert to 8-bit image for folium
    img_uint8 = (colors * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_b64}"

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Pipeline Controls")
    uploaded_file = st.file_uploader("Upload Mask", type=['png', 'jpg'])
    if uploaded_file:
        if st.button("Extract Topology"):
            raw_image = io.imread(uploaded_file, as_gray=True)
            mask = raw_image > 0.1
            skeleton = morphology.skeletonize(mask)
            G = sknw.build_sknw(skeleton)
            
            for u, v, d in G.edges(data=True): d['eff_weight'] = d['weight']
            for n in G.nodes: G.nodes[n]['o'] = G.nodes[n]['o'][::-1]
            
            st.session_state.graph = G
            st.session_state.mask = mask
            st.session_state.baseline_eff = calculate_network_efficiency(G)
            st.session_state.history = [1.0]

# --- MAIN DASHBOARD ---
if 'graph' in st.session_state:
    G = st.session_state.graph
    centrality_map = identify_gatekeepers(G)
    
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=15, tiles="CartoDB positron")
    
    image_b64 = mask_to_png_base64(st.session_state.mask, centrality_map, G)
    
    folium.raster_layers.ImageOverlay(
        image=image_b64,
        bounds=[[13.07, 80.26], [13.09, 80.28]],
        opacity=0.7
    ).add_to(m)
    
    st_folium(m, width=900, height=600)
    
    if st.button("Ablate Critical Node"):
        centrality = identify_gatekeepers(G)
        target = max(centrality, key=centrality.get)
        G.remove_node(target)
        new_eff = calculate_network_efficiency(G)
        st.session_state.history.append(new_eff / st.session_state.baseline_eff)
        st.rerun()

    st.line_chart(st.session_state.history)
