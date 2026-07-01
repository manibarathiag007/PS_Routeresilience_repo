import streamlit as st
import networkx as nx
import numpy as np
import folium
from streamlit_folium import st_folium
import copy

# =========================================================
# 0. INITIALIZE UI FIRST (Mandatory Streamlit Rule)
# =========================================================
st.set_page_config(page_title="Route Resilience Simulator", layout="wide")

# =========================================================
# 1. CORE PHYSICS & MATH ENGINE (From Phase III)
# =========================================================
def calculate_network_efficiency(graph, weight_attr='weight'):
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
                except KeyError:
                    pass
    return efficiency / (n * (n - 1))

def identify_gatekeepers(graph, weight_attr='weight'):
    """Calculates betweenness centrality to identify bottlenecks."""
    node_centrality = nx.betweenness_centrality(graph, weight=weight_attr)
    sorted_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)
    return sorted_nodes, node_centrality

# =========================================================
# 2. SESSION STATE MANAGEMENT
# =========================================================
if 'baseline_graph' not in st.session_state:
    # --- Generating a mock geospatial grid for demonstration ---
    G = nx.grid_2d_graph(5, 5)
    base_lat, base_lon = 13.0827, 80.2707
    for (i, j) in G.nodes():
        G.nodes[(i, j)]['pos'] = (base_lat + i*0.01, base_lon + j*0.01)
        
    for u, v in G.edges():
        G.edges[u, v]['weight'] = np.random.uniform(1.0, 5.0)
    
    G.add_edge((4, 4), (5, 5), weight=0.5) 
    G.nodes[(5, 5)]['pos'] = (base_lat + 0.05, base_lon + 0.05)
    # -----------------------------------------------------------
    
    st.session_state.baseline_graph = G
    st.session_state.current_graph = copy.deepcopy(G)
    st.session_state.baseline_eff = calculate_network_efficiency(G)
    st.session_state.history_r = [1.0] 
    st.session_state.nodes_removed = 0

# =========================================================
# 3. INTERACTIVE DASHBOARD UI
# =========================================================
st.title("🛰️ Route Resilience: Urban Topology Stress Test")
st.markdown("Bharatiya Antariksh Hackathon 2026 | **Simulating Network Percolation via Node Ablation**")

with st.sidebar:
    st.header("Simulation Controls")
    if st.button("🚨 Ablate Top Gatekeeper Node", use_container_width=True):
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
                
    if st.button("🔄 Reset Infrastructure", type="primary", use_container_width=True):
        st.session_state.current_graph = copy.deepcopy(st.session_state.baseline_graph)
        st.session_state.history_r = [1.0]
        st.session_state.nodes_removed = 0
        st.rerun()

col1, col2, col3 = st.columns(3)
current_r = st.session_state.history_r[-1]
col1.metric("Resilience Index (R)", f"{current_r:.3f}", f"{(current_r - 1.0)*100:.1f}%")
col2.metric("Nodes Ablated", st.session_state.nodes_removed)
col3.metric("Remaining Network Size", len(st.session_state.current_graph.nodes))

gatekeepers, centrality_map = identify_gatekeepers(st.session_state.current_graph)

all_lats = [data['pos'][0] for n, data in st.session_state.current_graph.nodes(data=True)]
all_lons = [data['pos'][1] for n, data in st.session_state.current_graph.nodes(data=True)]
if all_lats and all_lons:
    map_center = [np.mean(all_lats), np.mean(all_lons)]
else:
    map_center = [13.0827, 80.2707]

m = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB dark_matter")

for u, v, data in st.session_state.current_graph.edges(data=True):
    pos_u = st.session_state.current_graph.nodes[u]['pos']
    pos_v = st.session_state.current_graph.nodes[v]['pos']
    folium.PolyLine([pos_u, pos_v], color="#444444", weight=2, opacity=0.8).add_to(m)

for node, data in st.session_state.current_graph.nodes(data=True):
    c_score = centrality_map.get(node, 0)
    color = f"rgb({int(255 * c_score * 5)}, {100}, {int(255 * (1 - c_score * 5))})" 
    if c_score * 5 > 1: color = "red" 
    
    folium.CircleMarker(
        location=data['pos'],
        radius=5 + (c_score * 50), 
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
    st.caption("**Physics Overview:** This curve tracks the systemic degradation of the urban grid. A steep collapse indicates a highly fragile, centralized topology.")
