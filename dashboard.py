import streamlit as st
import networkx as nx
import numpy as np
import folium
from streamlit_folium import st_folium
import copy
from skimage import morphology, io
import sknw
import matplotlib.pyplot as plt
import itertools
import base64
import io
from PIL import Image
from scipy.ndimage import distance_transform_edt

# --- CONFIGURATION ---
st.set_page_config(page_title="Route Resilience Pipeline", layout="wide", page_icon="🛰️")

# --- CORE PHYSICS & TOPOLOGY ENGINE ---
def get_safe_base64_overlay(binary_mask, centrality_map, graph):
    """Generates a robust base64 PNG string of the centrality heatmap."""
    try:
        h, w = binary_mask.shape
        nodes = list(graph.nodes)
        if not nodes: return None
        
        node_coords = np.array([graph.nodes[n]['o'] for n in nodes])
        centrality_vals = np.array([centrality_map.get(n, 0) for n in nodes])
        
        y_grid, x_grid = np.indices((h, w))
        dist_sq = (x_grid[:, :, np.newaxis] - node_coords[:, 1])**2 + \
                  (y_grid[:, :, np.newaxis] - node_coords[:, 0])**2
        nearest_node_idx = np.argmin(dist_sq, axis=2)
        pixel_centrality = centrality_vals[nearest_node_idx]
        
        # Normalize
        norm_centrality = (pixel_centrality - pixel_centrality.min()) / \
                          (pixel_centrality.max() - pixel_centrality.min() + 1e-9)
        
        cmap = plt.get_cmap('coolwarm')
        rgba_img = cmap(norm_centrality)
        rgba_img[:, :, 3] = binary_mask * 0.7 
        
        img = Image.fromarray((rgba_img * 255).astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None

def heal_islands(graph, distance_threshold, penalty_confidence=0.2):
    components = list(nx.connected_components(graph))
    if len(components) <= 1: return graph
    for comp_a, comp_b in itertools.combinations(components, 2):
        min_dist = float('inf')
        best_pair = None
        for u in comp_a:
            for v in comp_b:
                coord_u = graph.nodes[u]['o']
                coord_v = graph.nodes[v]['o']
                dist = np.linalg.norm(coord_u - coord_v)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (u, v)
        if best_pair and min_dist < distance_threshold:
            u, v = best_pair
            graph.add_edge(u, v, weight=min_dist, eff_weight=min_dist/penalty_confidence)
    return graph

def merge_close_junctions(graph, merge_threshold):
    undirected_graph = graph.to_undirected()
    nodes_to_merge = set()
    for u, v, data in undirected_graph.edges(data=True):
        if 'weight' in data and data['weight'] < merge_threshold:
            nodes_to_merge.add((u, v))
    for u, v in nodes_to_merge:
        if graph.has_node(u) and graph.has_node(v):
            graph = nx.contracted_nodes(graph, u, v, self_loops=False)
    return graph

def calculate_network_efficiency(graph, weight_attr='eff_weight'):
    n = len(graph)
    if n < 2: return 0.0
    efficiency = 0.0
    try:
        paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight_attr))
        for u in graph:
            for v in graph:
                if u != v:
                    dist = paths[u].get(v, 0)
                    if dist > 0: efficiency += 1.0 / dist
    except: return 0.0
    return efficiency / (n * (n - 1))

def identify_gatekeepers(graph, weight_attr='eff_weight'):
    try:
        node_centrality = nx.betweenness_centrality(graph, weight=weight_attr)
        sorted_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes, node_centrality
    except: return [], {}

def process_mask_to_graph(image_bytes, road_width_px, bounds=None):
    raw_image = io.imread(image_bytes, as_gray=True)
    binary_mask = raw_image > 0.1
    image_h, image_w = binary_mask.shape
    skeleton = morphology.skeletonize(morphology.remove_small_objects(binary_mask, 50))
    graph = sknw.build_sknw(skeleton)
    for u, v, data in graph.edges(data=True): data['eff_weight'] = data['weight']
    graph = heal_islands(graph, road_width_px * 20)
    graph = merge_close_junctions(graph, road_width_px * 1.5)
    
    # Projection
    for n, data in graph.nodes(data=True):
        row, col = data['o']
        if bounds:
            lat = bounds['tl_lat'] - (row / image_h) * (bounds['tl_lat'] - bounds['br_lat'])
            lon = bounds['tl_lon'] + (col / image_w) * (bounds['br_lon'] - bounds['tl_lon'])
            graph.nodes[n]['pos'] = (lat, lon)
        else:
            graph.nodes[n]['pos'] = (13.0827 - (row * 0.00005), 80.2707 + (col * 0.00005))
    return binary_mask, skeleton, graph

# --- DASHBOARD ---
st.title("🛰️ Route Resilience Pipeline")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Mask", type=['png', 'jpg'])
    road_width = st.slider("Road Width (px)", 5, 50, 15)
    
    if uploaded_file and st.button("Extract Topology"):
        mask, skel, G = process_mask_to_graph(uploaded_file, road_width)
        st.session_state.current_graph = G
        st.session_state.raw_mask = mask
        st.session_state.baseline_eff = calculate_network_efficiency(G)
        st.session_state.history_r = [1.0]
        st.session_state.nodes_removed = 0

if 'current_graph' in st.session_state:
    G = st.session_state.current_graph
    
    if st.button("🚨 Ablate Critical Node"):
        _, centrality = identify_gatekeepers(G)
        if centrality:
            target = max(centrality, key=centrality.get)
            G.remove_node(target)
            st.session_state.nodes_removed += 1
            eff = calculate_network_efficiency(G)
            st.session_state.history_r.append(eff / st.session_state.baseline_eff)

    # Render Map
    _, centrality_map = identify_gatekeepers(G)
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=14, tiles="CartoDB dark_matter")
    
    overlay = get_safe_base64_overlay(st.session_state.raw_mask, centrality_map, G)
    if overlay:
        folium.raster_layers.ImageOverlay(image=overlay, bounds=[[13.07, 80.26], [13.09, 80.28]], opacity=0.7).add_to(m)
    
    for u, v, data in G.edges(data=True):
        folium.PolyLine([G.nodes[u]['pos'], G.nodes[v]['pos']], color="white", weight=1, opacity=0.3).add_to(m)
        
    st_folium(m, width=700, height=500)
    st.line_chart(st.session_state.history_r)
