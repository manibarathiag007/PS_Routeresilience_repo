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

# --- GRADIENT GENERATOR ---

def get_safe_base64_overlay(binary_mask, centrality_map, graph):
    """Safely converts map overlay to base64 for Folium."""
    h, w = binary_mask.shape
    nodes = list(graph.nodes)
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
    
    # Convert to PIL then B64
    img = Image.fromarray((rgba_img * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def get_gradient_overlay(binary_mask, centrality_map, graph):
    """Projects graph node centrality onto the spatial road mask."""
    h, w = binary_mask.shape
    nodes = list(graph.nodes)
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
    colored_overlay = cmap(norm_centrality)
    colored_overlay[:, :, 3] = binary_mask * 0.7 
    return colored_overlay

# --- CORE PHYSICS & TOPOLOGY ENGINE ---
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
            eff_weight = min_dist / penalty_confidence
            graph.add_edge(u, v, weight=min_dist, confidence=penalty_confidence, eff_weight=eff_weight)
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
    paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight_attr))
    for u in graph:
        for v in graph:
            if u != v:
                try:
                    dist = paths[u][v]
                    if dist > 0: efficiency += 1.0 / dist
                except KeyError: pass
    return efficiency / (n * (n - 1))

def identify_gatekeepers(graph, weight_attr='eff_weight'):
    node_centrality = nx.betweenness_centrality(graph, weight=weight_attr)
    sorted_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes, node_centrality

def process_mask_to_graph(image_bytes, road_width_px, bounds=None):
    raw_image = io.imread(image_bytes, as_gray=True)
    binary_mask = raw_image > 0.1
    image_h, image_w = binary_mask.shape
    noise_area = int((road_width_px ** 2) * 2)
    clean_mask = morphology.remove_small_objects(binary_mask, max_size=noise_area)
    clean_mask = morphology.remove_small_holes(clean_mask, max_size=noise_area)
    skeleton = morphology.skeletonize(clean_mask)
    graph = sknw.build_sknw(skeleton)
    for u, v, data in graph.edges(data=True):
        data['confidence'] = 1.0
        data['eff_weight'] = data['weight'] / data['confidence']
    graph = heal_islands(graph, distance_threshold=road_width_px * 20)
    graph = merge_close_junctions(graph, merge_threshold=road_width_px * 1.5)
    for n, data in graph.nodes(data=True):
        if 'o' in data:
            row, col = data['o']
            if bounds:
                lat_span = bounds['tl_lat'] - bounds['br_lat']
                lon_span = bounds['br_lon'] - bounds['tl_lon']
                lat = bounds['tl_lat'] - (row / image_h) * lat_span
                lon = bounds['tl_lon'] + (col / image_w) * lon_span
                graph.nodes[n]['pos'] = (lat, lon)
            else:
                base_lat, base_lon = 13.0827, 80.2707 
                scale_factor = 0.00005 
                graph.nodes[n]['pos'] = (base_lat - (row * scale_factor), base_lon + (col * scale_factor))
    return binary_mask, skeleton, graph

# --- MAIN DASHBOARD UI ---
st.set_page_config(page_title="Route Resilience Pipeline", layout="wide", page_icon="🛰️")
st.title("🛰️ Route Resilience: End-to-End Extraction & Stress Test")

with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Binary Mask (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    road_width = st.slider("Estimated Road Width (px)", min_value=5, max_value=50, value=15)
    use_real_bounds = st.checkbox("Use Real Geographic Bounds")
    bounds_dict = None
    if use_real_bounds:
        tl_lat = st.number_input("Top-Left Lat", value=13.09000, format="%.5f")
        tl_lon = st.number_input("Top-Left Lon", value=80.26000, format="%.5f")
        br_lat = st.number_input("Bottom-Right Lat", value=13.07000, format="%.5f")
        br_lon = st.number_input("Bottom-Right Lon", value=80.29000, format="%.5f")
        bounds_dict = {'tl_lat': tl_lat, 'tl_lon': tl_lon, 'br_lat': br_lat, 'br_lon': br_lon}
    
    if uploaded_file is not None:
        current_state_id = f"{uploaded_file.name}_{use_real_bounds}_{bounds_dict}"
        if st.session_state.get('last_run_state') != current_state_id:
            mask, skeleton, G = process_mask_to_graph(uploaded_file, road_width, bounds=bounds_dict)
            st.session_state.baseline_graph = G
            st.session_state.current_graph = copy.deepcopy(G)
            st.session_state.baseline_eff = calculate_network_efficiency(G)
            st.session_state.history_r = [1.0] 
            st.session_state.nodes_removed = 0
            st.session_state.last_run_state = current_state_id
            st.session_state.raw_mask = mask
            st.session_state.skeleton = skeleton

    if st.button("🚨 Ablate Top Gatekeeper", use_container_width=True) and 'current_graph' in st.session_state:
        G_sim = st.session_state.current_graph
        gatekeepers, _ = identify_gatekeepers(G_sim)
        if gatekeepers:
            G_sim.remove_node(gatekeepers[0][0])
            st.session_state.nodes_removed += 1
            new_eff = calculate_network_efficiency(G_sim)
            st.session_state.history_r.append(new_eff / st.session_state.baseline_eff)

if 'current_graph' not in st.session_state:
    st.info("Please upload a mask in the sidebar.")
else:
    # Safe retrieval
    G = st.session_state.current_graph
    mask = st.session_state.raw_mask
    _, centrality_map = identify_gatekeepers(G)
    
    # Define bounds (Safe defaults)
    bounds = [[13.07, 80.26], [13.09, 80.28]]
    
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=14, tiles="CartoDB dark_matter")
    
    # Add Overlay using the safe serialization
    b64_overlay = get_safe_base64_overlay(mask, centrality_map, G)
    folium.raster_layers.ImageOverlay(image=b64_overlay, bounds=bounds, opacity=0.7).add_to(m)
    
    # Render
    st_folium(m, width=700, height=500)
