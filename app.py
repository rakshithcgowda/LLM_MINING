import streamlit as st
from phase2 import OncologyLiteratureMiner
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import os

st.set_page_config(
    page_title="Oncology Literature Miner",
    page_icon=":microscope:",
    layout="wide"
)

@st.cache_resource
def get_miner():
    return OncologyLiteratureMiner()

miner = get_miner()

st.title("Oncology Literature Miner")
st.markdown("""
This tool extracts knowledge from oncology literature, clinical trials, and patents to build a 
comprehensive knowledge graph of cancer treatments, biomarkers, and their relationships.
""")

# Sidebar
with st.sidebar:
    st.header("Search Parameters")
    
    with st.expander("PubMed Search"):
        pubmed_query = st.text_input("PubMed Query", value="cancer AND (immunotherapy OR targeted therapy)")
        pubmed_max_results = st.slider("Max PubMed Results", 10, 1000, 100, 10)
        run_pubmed = st.button("Run PubMed Search")
    
    with st.expander("Clinical Trials"):
        trial_condition = st.text_input("Condition", value="breast cancer")
        trial_intervention = st.text_input("Intervention (optional)", value="")
        trial_max_results = st.slider("Max Trial Results", 5, 200, 50, 5)
        run_trials = st.button("Search Clinical Trials")
    
    with st.expander("Patents"):
        patent_query = st.text_input("Patent Search Query", value="cancer immunotherapy")
        patent_max_results = st.slider("Max Patent Results", 5, 100, 20, 5)
        run_patents = st.button("Search Patents")
    
    with st.expander("Visualization Settings"):
        max_nodes = st.slider("Max Nodes to Display", 10, 200, 50, 5)
        update_vis = st.button("Update Visualization")

    # Cancer type filter
    st.divider()
    st.subheader("Filter Cancer Types")
    all_cancers = sorted([
        node for node, attr in miner.graph.node_attributes.items()
        if attr.get('type') == 'cancer_type'
    ])
    selected_cancers = st.multiselect(
        "Select Cancer Types to Show",
        options=all_cancers,
        default=all_cancers
    )

# ----------- DEFINE TABS HERE -----------
tab1, tab2, tab3, tab4 = st.tabs(["Knowledge Graph", "Network Analysis", "Data Tables", "Export Data"])

# Trigger pipelines
if run_pubmed:
    with st.spinner("Searching PubMed..."):
        miner.run_pubmed_pipeline(query=pubmed_query, max_results=pubmed_max_results)
    st.success("PubMed done!")

if run_trials:
    with st.spinner("Searching Clinical Trials..."):
        miner.run_clinical_trials_pipeline(
            condition=trial_condition,
            intervention=trial_intervention or None,
            max_results=trial_max_results
        )
    st.success("Trials done!")

if run_patents:
    with st.spinner("Searching Patents..."):
        miner.run_patents_pipeline(query=patent_query, max_results=patent_max_results)
    st.success("Patents done!")

# ------------------- TAB 1: Knowledge Graph -------------------
with tab1:
    st.header("Interactive Knowledge Graph")
    if miner.graph.nodes:
        with st.spinner("Generating graph..."):
            nt = Network(
                height="600px",
                width="100%",
                bgcolor="#fff",
                font_color="black",
                directed=True,
                notebook=False,
                cdn_resources="remote"
            )
            G_full = miner.graph.to_networkx()

            # Filter by selected cancer types
            kept = [n for n, d in G_full.nodes(data=True)
                    if d.get('type') != 'cancer_type' or n in selected_cancers]
            G_filt = G_full.subgraph(kept).copy()

            # Degree pruning
            deg = dict(G_filt.degree())
            if len(deg) > max_nodes:
                top = sorted(deg, key=deg.get, reverse=True)[:max_nodes]
                G = G_filt.subgraph(top)
            else:
                G = G_filt

            # Add nodes
            for n, data in G.nodes(data=True):
                nt.add_node(
                    n,
                    label=n,
                    title=f"Type: {data.get('type')}",
                    color=data.get('color'),
                    size=10 + deg.get(n,1)*2
                )

            # Add edges
            for u, v, data in G.edges(data=True):
                nt.add_edge(
                    u, v,
                    title=data.get('type'),
                    label=data.get('type'),
                    width=1 + data.get('score',0.5)*2
                )

            # --- Physics settings for clarity ---
            nt.barnes_hut(
                gravity=-40000,
                central_gravity=0.1,
                spring_length=250,
                spring_strength=0.01,
                damping=0.9
            )
            nt.toggle_physics(True)
            nt.show_buttons(filter_=['physics'])

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            tmp.close()
            nt.save_graph(tmp.name)
            st.components.v1.html(open(tmp.name).read(), height=600, scrolling=True)
            os.unlink(tmp.name)
    else:
        st.info("No graph yet. Run a search first.")

# ------------------- TAB 2: Network Analysis -------------------
with tab2:
    st.header("Network Analysis")
    # ... your network analysis code ...

# ------------------- TAB 3: Data Tables -------------------
with tab3:
    st.header("Data Tables")
    # ... your data tables code ...

# ------------------- TAB 4: Export Data -------------------
with tab4:
    st.header("Export Data")
    # ... your export code ...
