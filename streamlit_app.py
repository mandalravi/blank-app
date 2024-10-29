import streamlit as st
import pandas as pd
import numpy as np
from pyvis.network import Network
from datetime import datetime, timedelta
import random
import os
import tempfile
import webbrowser
from pathlib import Path
import base64

# Set page config
st.set_page_config(
    page_title="Clinical Trial Network Visualizer",
    page_icon="🔬",
    layout="wide"
)

# Generate sample SDTM data
def create_dm_dataset(n_subjects=20):
    np.random.seed(42)
    dm_data = {
        'USUBJID': [f'SUBJ{str(i).zfill(3)}' for i in range(1, n_subjects + 1)],
        'AGE': np.random.randint(18, 75, n_subjects),
        'SEX': np.random.choice(['M', 'F'], n_subjects),
        'RACE': np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'OTHER'], n_subjects),
        'ARMCD': np.random.choice(['ARM1', 'ARM2', 'ARM3'], n_subjects),
        'COUNTRY': np.random.choice(['USA', 'GBR', 'FRA', 'DEU', 'JPN'], n_subjects),
        'BMI': np.random.uniform(18.5, 35.0, n_subjects).round(1)
    }
    return pd.DataFrame(dm_data)

def create_ae_dataset(dm_df):
    ae_records = []
    ae_terms = ['HEADACHE', 'NAUSEA', 'FATIGUE', 'DIZZINESS', 'RASH', 
                'INSOMNIA', 'ANXIETY', 'PAIN', 'COUGH', 'FEVER']
    severities = ['MILD', 'MODERATE', 'SEVERE']
    relationships = ['RELATED', 'POSSIBLY_RELATED', 'NOT_RELATED']
    
    for usubjid in dm_df['USUBJID']:
        n_events = np.random.randint(0, 4)
        for _ in range(n_events):
            start_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(1, 90))
            ae_records.append({
                'USUBJID': usubjid,
                'AETERM': np.random.choice(ae_terms),
                'AESEV': np.random.choice(severities),
                'AEREL': np.random.choice(relationships),
                'AESTDTC': start_date.strftime('%Y-%m-%d'),
                'AEENDTC': (start_date + timedelta(days=np.random.randint(1, 30))).strftime('%Y-%m-%d')
            })
    
    return pd.DataFrame(ae_records)

def create_network_graph(dm_df, ae_df):
    """Create network graph with filtered data"""
    net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
    
    # Add subject nodes
    for _, row in dm_df.iterrows():
        node_label = (f"Subject: {row['USUBJID']}\n"
                     f"Age: {row['AGE']}\n"
                     f"Sex: {row['SEX']}\n"
                     f"Race: {row['RACE']}\n"
                     f"Arm: {row['ARMCD']}\n"
                     f"Country: {row['COUNTRY']}")
        net.add_node(row['USUBJID'], label=node_label, 
                    title=node_label,
                    color='#97c2fc',
                    size=20)
    
    # Add AE nodes and edges
    for _, row in ae_df.iterrows():
        ae_node_id = f"{row['AETERM']}_{row['AESEV']}_{row['AEREL']}"
        node_label = (f"{row['AETERM']}\n"
                     f"({row['AESEV']})\n"
                     f"{row['AEREL']}")
        
        # Add AE node if it doesn't exist
        if not any(ae_node_id == node['id'] for node in net.nodes):
            color = {'MILD': '#90EE90', 'MODERATE': '#FFD700', 
                    'SEVERE': '#FF6B6B'}[row['AESEV']]
            net.add_node(ae_node_id, label=node_label, 
                        title=node_label,
                        color=color,
                        size=15)
        
        # Add edge between subject and AE
        net.add_edge(row['USUBJID'], ae_node_id, 
                    title=f"Start: {row['AESTDTC']}\nEnd: {row['AEENDTC']}")
    
    # Set physics layout options
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 150}
      }
    }
    """)
    
    return net

def main():
    st.title("Clinical Trial Network Visualizer")
    
    # Initialize session state for data
    if 'dm_df' not in st.session_state:
        st.session_state.dm_df = create_dm_dataset(20)
    if 'ae_df' not in st.session_state:
        st.session_state.ae_df = create_ae_dataset(st.session_state.dm_df)
    
    # Create sidebar for filters
    st.sidebar.header("Filter Options")
    
    # Demographics Filters
    st.sidebar.subheader("Demographics Filters")
    
    # Age range filter
    age_min = int(st.session_state.dm_df['AGE'].min())
    age_max = int(st.session_state.dm_df['AGE'].max())
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max)
    )
    
    # Sex filter
    sex_options = st.session_state.dm_df['SEX'].unique().tolist()
    selected_sex = st.sidebar.multiselect(
        "Sex",
        options=sex_options,
        default=sex_options
    )
    
    # Race filter
    race_options = st.session_state.dm_df['RACE'].unique().tolist()
    selected_race = st.sidebar.multiselect(
        "Race",
        options=race_options,
        default=race_options
    )
    
    # Treatment arm filter
    arm_options = st.session_state.dm_df['ARMCD'].unique().tolist()
    selected_arm = st.sidebar.multiselect(
        "Treatment Arm",
        options=arm_options,
        default=arm_options
    )
    
    # Country filter
    country_options = st.session_state.dm_df['COUNTRY'].unique().tolist()
    selected_country = st.sidebar.multiselect(
        "Country",
        options=country_options,
        default=country_options
    )
    
    # Adverse Events Filters
    st.sidebar.subheader("Adverse Events Filters")
    
    # Severity filter
    severity_options = st.session_state.ae_df['AESEV'].unique().tolist()
    selected_severity = st.sidebar.multiselect(
        "Severity",
        options=severity_options,
        default=severity_options
    )
    
    # Relationship filter
    relationship_options = st.session_state.ae_df['AEREL'].unique().tolist()
    selected_relationship = st.sidebar.multiselect(
        "Relationship to Treatment",
        options=relationship_options,
        default=relationship_options
    )
    
    # Maximum events per subject
    max_events = st.sidebar.number_input(
        "Maximum Events per Subject",
        min_value=1,
        max_value=10,
        value=4
    )
    
    # Apply filters
    filtered_dm = st.session_state.dm_df[
        (st.session_state.dm_df['AGE'].between(age_range[0], age_range[1])) &
        (st.session_state.dm_df['SEX'].isin(selected_sex)) &
        (st.session_state.dm_df['RACE'].isin(selected_race)) &
        (st.session_state.dm_df['ARMCD'].isin(selected_arm)) &
        (st.session_state.dm_df['COUNTRY'].isin(selected_country))
    ]
    
    filtered_ae = st.session_state.ae_df[
        (st.session_state.ae_df['AESEV'].isin(selected_severity)) &
        (st.session_state.ae_df['AEREL'].isin(selected_relationship)) &
        (st.session_state.ae_df['USUBJID'].isin(filtered_dm['USUBJID']))
    ]
    
    # Limit events per subject
    filtered_ae = (filtered_ae.groupby('USUBJID')
                  .apply(lambda x: x.head(max_events))
                  .reset_index(drop=True))
    
    # Create two columns for metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Subjects", len(filtered_dm))
    with col2:
        st.metric("Number of Adverse Events", len(filtered_ae))
    
    # Create and display network
    if len(filtered_dm) > 0 and len(filtered_ae) > 0:
        net = create_network_graph(filtered_dm, filtered_ae)
        
        # Save the network to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
            net.save_graph(tmp_file.name)
            
            # Display the network in an iframe
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html_content = f.read()
                st.components.v1.html(html_content, height=800)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
    else:
        st.warning("No data available for the selected filters. Please adjust your selection.")
    
    # Show filtered data tables
    st.subheader("Filtered Demographics Data")
    st.dataframe(filtered_dm)
    
    st.subheader("Filtered Adverse Events Data")
    st.dataframe(filtered_ae)

if __name__ == "__main__":
    main()