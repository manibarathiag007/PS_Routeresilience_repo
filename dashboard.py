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

# =========================================================
# 0. INITIALIZE UI FIRST
# =========================================================
st.set_page_config(page_title="Route Resilience Pipeline", layout="wide", page_icon="🛰️")

# =========================================================
# 1. CORE PHYSICS & TOPOLOGY ENGINE (The "Accelerator")
# =========================================================
def heal_islands(graph, distance_threshold, penalty_confidence=0.2):
    """Bridges isolated subgraphs based on minimum Euclidean vectors."""
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
    """Contracts microscopic artifact nodes into single centers of mass."""
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
    """Calculates global efficiency: E = 1/(N(N-1)) * sum(1/d_uv)"""
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
    """Calculates betweenness centrality to identify bottlenecks."""
    node_centrality = nx.betweenness_centrality(graph, weight=weight_attr)
    sorted_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes, node_centrality

# =========================================================
# 2. DATA PIPELINE (Pixel Mask -> Geospatial Graph)
# =========================================================
def process_mask_to_graph(image_bytes, road_width_px, bounds=None):
    """Converts a raw image into a fully healed, geospatial NetworkX graph."""
    # 1. Load and Binarize
    raw_image = io.imread(image_bytes, as_gray=True)
    binary_mask = raw_image > 0.1
    image_h, image_w = binary_mask.shape
    
    # 2. Morphological Cleaning
    noise_area = int((road_width_px ** 2) * 2)
    clean_mask = morphology.remove_small_objects(binary_mask, max_size=noise_area)
    clean_mask = morphology.remove_small_holes(clean_mask, max_size=noise_area)
    
    # 3. Skeletonize and build graph
    skeleton = morphology.skeletonize(clean_mask)
    graph = sknw.build_sknw(skeleton)
    
    # Initialize basic weights
    for u, v, data in graph.edges(data=True):
        data['confidence'] = 1.0
        data['eff_weight'] = data['weight'] / data['confidence']
        
    # 4. Heal Topology
    graph = heal_islands(graph, distance_threshold=road_width_px * 20)
    graph = merge_close_junctions(graph, merge_threshold=road_width_px * 1.5)
    
    # 5. Coordinate Projection (Linear Space Transformation)
    for n, data in graph.nodes(data=True):
        if 'o' in data:
            row, col = data['o']
            
            if bounds:
                # Real geospatial mapping via linear interpolation
                lat_span = bounds['tl_lat'] - bounds['br_lat']
                lon_span = bounds['br_lon'] - bounds['tl_lon']
                
                # Row 0 is Top (North/Max Lat). Col 0 is Left (West/Min Lon)
                lat = bounds['tl_lat'] - (row / image_h) * lat_span
                lon = bounds['tl_lon'] + (col / image_w) * lon_span
                graph.nodes[n]['pos'] = (lat, lon)
            else:
                # Fallback: Pseudo-Geospatial mapping (Arbitrary scale near Chennai)
                base_lat, base_lon = 13.0827, 80.2707 
                scale_factor = 0.00005 
                graph.nodes[n]['pos'] = (base_lat - (row * scale_factor), base_lon + (col * scale_factor))
            
    return binary_mask, skeleton, graph

# =========================================================
# 3. INTERACTIVE DASHBOARD UI
# =========================================================
st.title("🛰️ Route Resilience: End-to-End Extraction & Stress Test")
st.markdown("Bharatiya Antariksh Hackathon 2026 | **From Pixel Mask to Percolation Phase Transition**")

# --- SIDEBAR: INPUT & CONTROLS ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload Binary Mask (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    road_width = st.slider("Estimated Road Width (px)", min_value=5, max_value=50, value=15)
    
    st.divider()
    st.header("2. Geospatial Alignment")
    use_real_bounds = st.checkbox("Use Real Geographic Bounds")
    bounds_dict = None
    
    if use_real_bounds:
        st.markdown("<small>Enter Bounding Box Coordinates (Dec. Degrees):</small>", unsafe_allow_html=True)
        tl_lat = st.number_input("Top-Left Lat (North)", value=13.09000, format="%.5f")
        tl_lon = st.number_input("Top-Left Lon (West)", value=80.26000, format="%.5f")
        br_lat = st.number_input("Bottom-Right Lat (South)", value=13.07000, format="%.5f")
        br_lon = st.number_input("Bottom-Right Lon (East)", value=80.29000, format="%.5f")
        bounds_dict = {'tl_lat': tl_lat, 'tl_lon': tl_lon, 'br_lat': br_lat, 'br_lon': br_lon}
    else:
        st.caption("Using arbitrary pseudo-coordinates for testing.")
    
    if uploaded_file is not None:
        # Check if file OR bounds changed to trigger re-run
        current_state_id = f"{uploaded_file.name}_{use_real_bounds}_{bounds_dict}"
        if st.session_state.get('last_run_state') != current_state_id:
            with st.spinner("Executing Mathematical Extraction..."):
                mask, skeleton, G = process_mask_to_graph(uploaded_file, road_width, bounds=bounds_dict)
                
                st.session_state.baseline_graph = G
                st.session_state.current_graph = copy.deepcopy(G)
                st.session_state.baseline_eff = calculate_network_efficiency(G)
                st.session_state.history_r = [1.0] 
                st.session_state.nodes_removed = 0
                st.session_state.last_run_state = current_state_id
                
                st.session_state.raw_mask = mask
                st.session_state.skeleton = skeleton

    st.divider()
    st.header("3. Simulation Controls")
    if st.button("🚨 Ablate Top Gatekeeper Node", use_container_width=True) and 'current_graph' in st.session_state:
        G_sim = st.session_state.current_graph
        if len(G_sim) > 1:
            gatekeepers, _ = identify_gatekeepers(G_sim)
            if gatekeepers:
                target_node = gatekeepers[0][0]
                G_sim.remove_node(target_node)
                st.session_state.nodes_removed += 1
                
                new_eff = calculate_network_efficiency(G_sim)
                new_r = new_eff / st.session_state.baseline_eff if st.session_state.baseline_eff > 0 else 0
                st.session_state.history_r.append(new_r)
                
    if st.button("🔄 Reset Infrastructure", type="primary", use_container_width=True) and 'baseline_graph' in st.session_state:
        st.session_state.current_graph = copy.deepcopy(st.session_state.baseline_graph)
        st.session_state.history_r = [1.0]
        st.session_state.nodes_removed = 0
        st.rerun()

# --- MAIN DASHBOARD (Only shows if file is uploaded) ---
if 'current_graph' in st.session_state:
    
    with st.expander("🔍 View Extraction Pipeline (Mask -> Skeleton -> Graph)", expanded=False):
        c1, c2 = st.columns(2)
        c1.image(st.session_state.raw_mask, caption="Raw Binary Mask", use_container_width=True)
        c2.image(st.session_state.skeleton, caption="Healed Topological Skeleton", use_container_width=True, clamp=True)
        
    col1, col2, col3 = st.columns(3)
    current_r = st.session_state.history_r[-1]
    col1.metric("Resilience Index (R)", f"{current_r:.3f}", f"{(current_r - 1.0)*100:.1f}%")
    col2.metric("Nodes Ablated (Destroyed)", st.session_state.nodes_removed)
    col3.metric("Remaining Network Size", len(st.session_state.current_graph.nodes))

    gatekeepers, centrality_map = identify_gatekeepers(st.session_state.current_graph)
    
    all_lats = [data['pos'][0] for n, data in st.session_state.current_graph.nodes(data=True) if 'pos' in data]
    all_lons = [data['pos'][1] for n, data in st.session_state.current_graph.nodes(data=True) if 'pos' in data]
    
    # Check if we have valid bounding coordinates to frame the map correctly
    if use_real_bounds and bounds_dict:
        # Fit bounds to the user-provided geospatial window
        map_center = [(bounds_dict['tl_lat'] + bounds_dict['br_lat'])/2, (bounds_dict['tl_lon'] + bounds_dict['br_lon'])/2]
    else:
        map_center = [np.mean(all_lats), np.mean(all_lons)] if all_lats else [13.0827, 80.2707]

    m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB dark_matter")
    
    # If using real bounds, draw a subtle bounding box to show the exact satellite footprint
    if use_real_bounds and bounds_dict:
        folium.Rectangle(
            bounds=[[bounds_dict['br_lat'], bounds_dict['tl_lon']], [bounds_dict['tl_lat'], bounds_dict['br_lon']]],
            color='#ffffff', weight=1, fill=False, opacity=0.3
        ).add_to(m)

    for u, v, data in st.session_state.current_graph.edges(data=True):
        if 'pos' in st.session_state.current_graph.nodes[u] and 'pos' in st.session_state.current_graph.nodes[v]:
            pos_u = st.session_state.current_graph.nodes[u]['pos']
            pos_v = st.session_state.current_graph.nodes[v]['pos']
            folium.PolyLine([pos_u, pos_v], color="#444444", weight=2, opacity=0.8).add_to(m)

    for node, data in st.session_state.current_graph.nodes(data=True):
        if 'pos' in data:
            c_score = centrality_map.get(node, 0)
            color = f"rgb({int(255 * c_score * 5)}, {100}, {int(255 * (1 - c_score * 5))})" 
            if c_score * 5 > 1: color = "red" 
            
            folium.CircleMarker(
                location=data['pos'],
                radius=3 + (c_score * 30), 
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=f"Node: {node}<br>Centrality: {c_score:.4f}"
            ).add_to(m)

    col_map, col_chart = st.columns([2, 1])
    with col_map:
        st.markdown("### Structural Intelligence Heatmap")
        st_folium(m, width=700, height=500, returned_objects=[])
    with col_chart:
        st.markdown("### Percolation Phase Transition")
        st.line_chart(st.session_state.history_r, height=400, y_label="Resilience Index (R)")
else:
    st.info("👈 Please upload a binary mask in the sidebar to initiate the extraction pipeline.")
