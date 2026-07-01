import streamlit as st
import networkx as nx
import numpy as np
import folium
from streamlit_folium import st_folium
import copy
from skimage import morphology, io, color
import sknw
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools

st.set_page_config(page_title="Route Resilience Pipeline", layout="wide", page_icon="🛰️")

# --- CORE PHYSICS ENGINE ---
def calculate_network_efficiency(graph, weight_attr='eff_weight'):
    n = len(graph)
    if n < 2: return 0.0
    efficiency = 0.0
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
    # Normalize betweenness centrality to [0, 1] range
    centrality = nx.betweenness_centrality(graph, weight='eff_weight')
    return centrality

def mask_to_colored_overlay(binary_mask, centrality_map, graph):
    """Maps node centrality to the road pixels using a red-blue gradient."""
    h, w = binary_mask.shape
    colored_mask = np.zeros((h, w, 4))
    
    # Create a blank canvas
    cmap = plt.get_cmap('coolwarm')
    
    # Heuristic: assign each pixel to the nearest node's centrality score
    # For a faster implementation, we use distance transform
    from scipy.ndimage import distance_transform_edt
    
    node_coords = np.array([graph.nodes[n]['o'] for n in graph.nodes])
    grid_y, grid_x = np.indices((h, w))
    
    # Map centrality values to nodes
    vals = np.array([centrality_map.get(n, 0) for n in graph.nodes])
    
    # Assign each pixel color based on the nearest node
    dist_sq = (grid_x[:, :, np.newaxis] - node_coords[:, 1])**2 + (grid_y[:, :, np.newaxis] - node_coords[:, 0])**2
    nearest_node_idx = np.argmin(dist_sq, axis=2)
    
    pixel_centrality = vals[nearest_node_idx]
    
    # Apply colormap
    colors = cmap(pixel_centrality * 5) # Scale factor for visibility
    colored_mask = colors
    colored_mask[:, :, 3] = binary_mask * 0.6  # Apply transparency to road pixels
    
    return colored_mask

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
            
            # Setup graph weights
            for u, v, d in G.edges(data=True): d['eff_weight'] = d['weight']
            for n in G.nodes: G.nodes[n]['o'] = G.nodes[n]['o'][::-1] # Fix sknw axis
            
            st.session_state.graph = G
            st.session_state.mask = mask
            st.session_state.baseline_eff = calculate_network_efficiency(G)
            st.session_state.history = [1.0]

# --- MAIN DASHBOARD ---
if 'graph' in st.session_state:
    G = st.session_state.graph
    centrality_map = identify_gatekeepers(G)
    
    # Display Map
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=15, tiles="CartoDB positron")
    
    # Generate Colored Mask
    overlay = mask_to_colored_overlay(st.session_state.mask, centrality_map, G)
    
    # Add ImageOverlay
    folium.raster_layers.ImageOverlay(
        image=overlay,
        bounds=[[13.07, 80.26], [13.09, 80.28]],
        opacity=0.7
    ).add_to(m)
    
    st_folium(m, width=900, height=600)
    
    # Simulation Logic
    if st.button("Ablate Critical Node"):
        centrality = identify_gatekeepers(G)
        target = max(centrality, key=centrality.get)
        G.remove_node(target)
        new_eff = calculate_network_efficiency(G)
        st.session_state.history.append(new_eff / st.session_state.baseline_eff)
        st.rerun()

    st.line_chart(st.session_state.history)
