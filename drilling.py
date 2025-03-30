import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import base64
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Analyse de forages miniers",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour télécharger les données en CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'Télécharger {text}'
    return href

# Fonction pour créer une représentation 3D des forages
def create_drillhole_3d_plot(collars_df, survey_df, lithology_df=None, assays_df=None, 
                            hole_id_col=None, x_col=None, y_col=None, z_col=None,
                            azimuth_col=None, dip_col=None, depth_col=None,
                            lith_from_col=None, lith_to_col=None, lith_col=None,
                            assay_from_col=None, assay_to_col=None, assay_value_col=None):
    
    if collars_df is None or survey_df is None or hole_id_col is None or x_col is None or y_col is None or z_col is None:
        st.error("Données insuffisantes pour créer une visualisation 3D")
        return None
    
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    
    # Pour chaque trou de forage
    for hole_id in collars_df[hole_id_col].unique():
        # Récupérer les données de collar pour ce trou
        collar = collars_df[collars_df[hole_id_col] == hole_id]
        if collar.empty:
            continue
            
        # Point de départ du trou (collar)
        x_start = collar[x_col].values[0]
        y_start = collar[y_col].values[0]
        z_start = collar[z_col].values[0]
        
        # Récupérer les données de survey pour ce trou
        hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        if hole_surveys.empty:
            continue
            
        # Calculer les points 3D pour le tracé du trou
        x_points = [x_start]
        y_points = [y_start]
        z_points = [z_start]
        
        current_x, current_y, current_z = x_start, y_start, z_start
        prev_depth = 0
        
        for _, survey in hole_surveys.iterrows():
            depth = survey[depth_col]
            azimuth = survey[azimuth_col]
            dip = survey[dip_col]
            
            segment_length = depth - prev_depth
            
            # Convertir l'azimuth et le dip en direction 3D
            azimuth_rad = np.radians(azimuth)
            dip_rad = np.radians(dip)
            
            # Calculer la nouvelle position
            dx = segment_length * np.sin(dip_rad) * np.sin(azimuth_rad)
            dy = segment_length * np.sin(dip_rad) * np.cos(azimuth_rad)
            dz = -segment_length * np.cos(dip_rad)  # Z est négatif pour la profondeur
            
            current_x += dx
            current_y += dy
            current_z += dz
            
            x_points.append(current_x)
            y_points.append(current_y)
            z_points.append(current_z)
            
            prev_depth = depth
        
        # Ajouter la trace du trou de forage
        fig.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=z_points,
                mode='lines',
                name=f'Forage {hole_id}',
                line=dict(width=4, color='blue'),
                hoverinfo='text',
                hovertext=[f'ID: {hole_id}
X: {x:.2f}
Y: {y:.2f}
Z: {z:.2f}' 
                           for x, y, z in zip(x_points, y_points, z_points)]
            )
        )
        
        # Ajouter les intersections lithologiques si disponibles
        if lithology_df is not None and lith_from_col is not None and lith_to_col is not None and lith_col is not None:
            hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id]
            
            if not hole_litho.empty:
                # Pour chaque segment de lithologie
                for _, litho in hole_litho.iterrows():
                    from_depth = litho[lith_from_col]
                    to_depth = litho[lith_to_col]
                    lith_type = litho[lith_col]
                    
                    # Trouver les coordonnées 3D pour ce segment
                    # (calcul simplifié - interpolation linéaire)
                    litho_x, litho_y, litho_z = [], [], []
                    
                    for i in range(len(hole_surveys) - 1):
                        depth1 = hole_surveys.iloc[i][depth_col]
                        depth2 = hole_surveys.iloc[i + 1][depth_col]
                        
                        # Si le segment est dans cette plage de profondeur
                        if (from_depth >= depth1 and from_depth < depth2) or (to_depth > depth1 and to_depth <= depth2) or (from_depth <= depth1 and to_depth >= depth2):
                            # Interpolation des points pour ce segment
                            segment_points = min(10, int((depth2 - depth1) / 0.1))  # Nombre de points à interpoler
                            for j in range(segment_points):
                                interp_depth = depth1 + (depth2 - depth1) * j / segment_points
                                if from_depth <= interp_depth <= to_depth:
                                    # Interpolation des coordonnées
                                    idx = i + j / segment_points
                                    pos = min(len(x_points) - 2, int(idx))
                                    frac = idx - pos
                                    
                                    interp_x = x_points[pos] + frac * (x_points[pos + 1] - x_points[pos])
                                    interp_y = y_points[pos] + frac * (y_points[pos + 1] - y_points[pos])
                                    interp_z = z_points[pos] + frac * (z_points[pos + 1] - z_points[pos])
                                    
                                    litho_x.append(interp_x)
                                    litho_y.append(interp_y)
                                    litho_z.append(interp_z)
                    
                    if litho_x:
                        # Couleur basée sur le type de lithologie (simplifiée)
                        color = px.colors.qualitative.Plotly[hash(lith_type) % len(px.colors.qualitative.Plotly)]
                        
                        fig.add_trace(
                            go.Scatter3d(
                                x=litho_x,
                                y=litho_y,
                                z=litho_z,
                                mode='markers',
                                name=f'Lithologie: {lith_type}',
                                marker=dict(size=5, color=color),
                                hoverinfo='text',
                                hovertext=[f'Lithologie: {lith_type}
Profondeur: {from_depth:.2f}-{to_depth:.2f}m' 
                                          for _ in litho_x]
                            )
                        )
        
        # Ajouter les valeurs d'analyses si disponibles
        if assays_df is not None and assay_from_col is not None and assay_to_col is not None and assay_value_col is not None:
            hole_assays = assays_df[assays_df[hole_id_col] == hole_id]
            
            if not hole_assays.empty:
                # Pour chaque segment d'analyse
                for _, assay in hole_assays.iterrows():
                    from_depth = assay[assay_from_col]
                    to_depth = assay[assay_to_col]
                    value = assay[assay_value_col]
                    
                    # Même logique d'interpolation que pour la lithologie
                    assay_x, assay_y, assay_z = [], [], []
                    
                    for i in range(len(hole_surveys) - 1):
                        depth1 = hole_surveys.iloc[i][depth_col]
                        depth2 = hole_surveys.iloc[i + 1][depth_col]
                        
                        if (from_depth >= depth1 and from_depth < depth2) or (to_depth > depth1 and to_depth <= depth2) or (from_depth <= depth1 and to_depth >= depth2):
                            segment_points = min(5, int((depth2 - depth1) / 0.2))
                            for j in range(segment_points):
                                interp_depth = depth1 + (depth2 - depth1) * j / segment_points
                                if from_depth <= interp_depth <= to_depth:
                                    idx = i + j / segment_points
                                    pos = min(len(x_points) - 2, int(idx))
                                    frac = idx - pos
                                    
                                    interp_x = x_points[pos] + frac * (x_points[pos + 1] - x_points[pos])
                                    interp_y = y_points[pos] + frac * (y_points[pos + 1] - y_points[pos])
                                    interp_z = z_points[pos] + frac * (z_points[pos + 1] - z_points[pos])
                                    
                                    assay_x.append(interp_x)
                                    assay_y.append(interp_y)
                                    assay_z.append(interp_z)
                    
                    if assay_x:
                        # Couleur basée sur la valeur (échelle de rouge)
                        # Normaliser la valeur pour la colorimétrie
                        max_value = assays_df[assay_value_col].max()
                        normalized_value = min(1.0, value / max_value if max_value > 0 else 0)
                        
                        fig.add_trace(
                            go.Scatter3d(
                                x=assay_x,
                                y=assay_y,
                                z=assay_z,
                                mode='markers',
                                name=f'Teneur: {value:.2f}',
                                marker=dict(
                                    size=6, 
                                    color=value,
                                    colorscale='Reds',
                                    colorbar=dict(title=assay_value_col)
                                ),
                                hoverinfo='text',
                                hovertext=[f'Teneur: {value:.2f}
Profondeur: {from_depth:.2f}-{to_depth:.2f}m' 
                                          for _ in assay_x]
                            )
                        )
    
    # Ajuster la mise en page
    fig.update_layout(
        title="Visualisation 3D des forages",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (Élévation)",
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

# Titre de l'application
st.title('Analyse de données de forages miniers')

# Barre latérale pour la navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Sélectionnez une page:', [
    'Chargement des données', 
    'Aperçu des données', 
    'Statistiques', 
    'Visualisation 3D'
])

# Initialisation des variables de session
if 'collars_df' not in st.session_state:
    st.session_state.collars_df = None
    
if 'survey_df' not in st.session_state:
    st.session_state.survey_df = None
    
if 'lithology_df' not in st.session_state:
    st.session_state.lithology_df = None
    
if 'assays_df' not in st.session_state:
    st.session_state.assays_df = None

if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'hole_id': None,
        'x': None,
        'y': None,
        'z': None,
        'azimuth': None,
        'dip': None,
        'depth': None,
        'lith_from': None,
        'lith_to': None,
        'lithology': None,
        'assay_from': None,
        'assay_to': None,
        'assay_value': None
    }

# Page de chargement des données
if page == 'Chargement des données':
    st.header('Chargement des données')
    
    # Créer des onglets pour les différents types de données
    tabs = st.tabs(["Collars", "Survey", "Lithology", "Assays"])
    
    # Onglet Collars
    with tabs[0]:
        st.subheader('Chargement des données de collars')
        
        collars_file = st.file_uploader("Télécharger le fichier CSV des collars", type=['csv'])
        if collars_file is not None:
            st.session_state.collars_df = pd.read_csv(collars_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.collars_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.collars_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou", [''] + cols, 
                                     index=cols.index(st.session_state.column_mapping['hole_id']) + 1 if st.session_state.column_mapping['hole_id'] in cols else 0)
            with col2:
                st.session_state.column_mapping['x'] = st.selectbox("Colonne X", [''] + cols,
                                index=cols.index(st.session_state.column_mapping['x']) + 1 if st.session_state.column_mapping['x'] in cols else 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['y'] = st.selectbox("Colonne Y", [''] + cols,
                                index=cols.index(st.session_state.column_mapping['y']) + 1 if st.session_state.column_mapping['y'] in cols else 0)
            with col2:
                st.session_state.column_mapping['z'] = st.selectbox("Colonne Z", [''] + cols,
                                index=cols.index(st.session_state.column_mapping['z']) + 1 if st.session_state.column_mapping['z'] in cols else 0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données"):
                st.dataframe(st.session_state.collars_df.head())
    
    # Onglet Survey
    with tabs[1]:
        st.subheader('Chargement des données de survey')
        
        survey_file = st.file_uploader("Télécharger le fichier CSV des surveys", type=['csv'])
        if survey_file is not None:
            st.session_state.survey_df = pd.read_csv(survey_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.survey_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.survey_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Survey)", [''] + cols, 
                                     index=cols.index(st.session_state.column_mapping['hole_id']) + 1 if st.session_state.column_mapping['hole_id'] in cols else 0)
            with col2:
                st.session_state.column_mapping['depth'] = st.selectbox("Colonne profondeur", [''] + cols,
                                  index=cols.index(st.session_state.column_mapping['depth']) + 1 if st.session_state.column_mapping['depth'] in cols else 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['azimuth'] = st.selectbox("Colonne azimut", [''] + cols,
                                    index=cols.index(st.session_state.column_mapping['azimuth']) + 1 if st.session_state.column_mapping['azimuth'] in cols else 0)
            with col2:
                st.session_state.column_mapping['dip'] = st.selectbox("Colonne pendage", [''] + cols,
                                index=cols.index(st.session_state.column_mapping['dip']) + 1 if st.session_state.column_mapping['dip'] in cols else 0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Survey)"):
                st.dataframe(st.session_state.survey_df.head())
    
    # Onglet Lithology
    with tabs[2]:
        st.subheader('Chargement des données de lithologie')
        
        lithology_file = st.file_uploader("Télécharger le fichier CSV des lithologies", type=['csv'])
        if lithology_file is not None:
            st.session_state.lithology_df = pd.read_csv(lithology_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.lithology_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.lithology_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Lithology)", [''] + cols, 
                                     index=cols.index(st.session_state.column_mapping['hole_id']) + 1 if st.session_state.column_mapping['hole_id'] in cols else 0)
            with col2:
                st.session_state.column_mapping['lithology'] = st.selectbox("Colonne de lithologie", [''] + cols,
                                     index=cols.index(st.session_state.column_mapping['lithology']) + 1 if st.session_state.column_mapping['lithology'] in cols else 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['lith_from'] = st.selectbox("Colonne de profondeur début", [''] + cols,
                                     index=cols.index(st.session_state.column_mapping['lith_from']) + 1 if st.session_state.column_mapping['lith_from'] in cols else 0)
            with col2:
                st.session_state.column_mapping['lith_to'] = st.selectbox("Colonne de profondeur fin", [''] + cols,
                                   index=cols.index(st.session_state.column_mapping['lith_to']) + 1 if st.session_state.column_mapping['lith_to'] in cols else 0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Lithology)"):
                st.dataframe(st.session_state.lithology_df.head())
    
    # Onglet Assays
    with tabs[3]:
        st.subheader('Chargement des données d\'analyses')
        
        assays_file = st.file_uploader("Télécharger le fichier CSV des analyses", type=['csv'])
        if assays_file is not None:
            st.session_state.assays_df = pd.read_csv(assays_file)
            st.success(f"Fichier chargé avec succès. {len(st.session_state.assays_df)} enregistrements trouvés.")
            
            # Sélection des colonnes importantes
            st.subheader("Sélection des colonnes")
            cols = st.session_state.assays_df.columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['hole_id'] = st.selectbox("Colonne ID du trou (Assays)", [''] + cols, 
                                     index=cols.index(st.session_state.column_mapping['hole_id']) + 1 if st.session_state.column_mapping['hole_id'] in cols else 0)
            with col2:
                st.session_state.column_mapping['assay_value'] = st.selectbox("Colonne de valeur (par ex. Au g/t)", [''] + cols,
                                       index=cols.index(st.session_state.column_mapping['assay_value']) + 1 if st.session_state.column_mapping['assay_value'] in cols else 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.column_mapping['assay_from'] = st.selectbox("Colonne de profondeur début (Assays)", [''] + cols,
                                       index=cols.index(st.session_state.column_mapping['assay_from']) + 1 if st.session_state.column_mapping['assay_from'] in cols else 0)
            with col2:
                st.session_state.column_mapping['assay_to'] = st.selectbox("Colonne de profondeur fin (Assays)", [''] + cols,
                                     index=cols.index(st.session_state.column_mapping['assay_to']) + 1 if st.session_state.column_mapping['assay_to'] in cols else 0)
            
            # Aperçu des données
            if st.checkbox("Afficher l'aperçu des données (Assays)"):
                st.dataframe(st.session_state.assays_df.head())

# Page d'aperçu des données
elif page == 'Aperçu des données':
    st.header('Aperçu des données')
    
    # Vérifier si des données ont été chargées
    if st.session_state.collars_df is None and st.session_state.survey_df is None and st.session_state.lithology_df is None and st.session_state.assays_df is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger des données.")
    else:
        # Créer des onglets pour les différents types de données
        data_tabs = st.tabs(["Collars", "Survey", "Lithology", "Assays"])
        
        # Onglet Collars
        with data_tabs[0]:
            if st.session_state.collars_df is not None:
                st.subheader('Données de collars')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.collars_df)}")
                st.dataframe(st.session_state.collars_df)
                
                st.markdown(get_csv_download_link(st.session_state.collars_df, "collars_data.csv", "les données de collars"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de collars n'a été chargée.")
        
        # Onglet Survey
        with data_tabs[1]:
            if st.session_state.survey_df is not None:
                st.subheader('Données de survey')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.survey_df)}")
                st.dataframe(st.session_state.survey_df)
                
                st.markdown(get_csv_download_link(st.session_state.survey_df, "survey_data.csv", "les données de survey"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de survey n'a été chargée.")
        
        # Onglet Lithology
        with data_tabs[2]:
            if st.session_state.lithology_df is not None:
                st.subheader('Données de lithologie')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.lithology_df)}")
                st.dataframe(st.session_state.lithology_df)
                
                st.markdown(get_csv_download_link(st.session_state.lithology_df, "lithology_data.csv", "les données de lithologie"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée de lithologie n'a été chargée.")
        
        # Onglet Assays
        with data_tabs[3]:
            if st.session_state.assays_df is not None:
                st.subheader('Données d\'analyses')
                st.write(f"Nombre total d'enregistrements: {len(st.session_state.assays_df)}")
                st.dataframe(st.session_state.assays_df)
                
                st.markdown(get_csv_download_link(st.session_state.assays_df, "assays_data.csv", "les données d'analyses"), unsafe_allow_html=True)
            else:
                st.info("Aucune donnée d'analyses n'a été chargée.")

# Page de statistiques
elif page == 'Statistiques':
    st.header('Statistiques')
    
    # Vérifier si des données ont été chargées
    if st.session_state.collars_df is None and st.session_state.survey_df is None and st.session_state.lithology_df is None and st.session_state.assays_df is None:
        st.warning("Aucune donnée n'a été chargée. Veuillez d'abord charger des données.")
    else:
        # Créer des onglets pour les différents types de statistiques
        stats_tabs = st.tabs(["Résumé général", "Lithologie", "Analyses"])
        
        # Onglet Résumé général
        with stats_tabs[0]:
            st.subheader('Résumé général des forages')
            
            # Statistiques sur les forages
            if st.session_state.collars_df is not None and st.session_state.column_mapping['hole_id']:
                hole_id_col = st.session_state.column_mapping['hole_id']
                
                num_holes = st.session_state.collars_df[hole_id_col].nunique()
                st.write(f"Nombre total de forages: {num_holes}")
                
                # Profondeur totale des forages
                if st.session_state.survey_df is not None and st.session_state.column_mapping['depth']:
                    depth_col = st.session_state.column_mapping['depth']
                    max_depths = st.session_state.survey_df.groupby(hole_id_col)[depth_col].max()
                    total_depth = max_depths.sum()
                    avg_depth = max_depths.mean()
                    max_depth = max_depths.max()
                    
                    st.write(f"Profondeur totale forée: {total_depth:.2f} m")
                    st.write(f"Profondeur moyenne des forages: {avg_depth:.2f} m")
                    st.write(f"Profondeur maximale: {max_depth:.2f} m")
                    
                    # Histogramme des profondeurs
                    fig = px.histogram(max_depths, 
                                       title="Distribution des profondeurs de forage",
                                       labels={'value': 'Profondeur (m)', 'count': 'Nombre de forages'},
                                       nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Données de collars incomplètes ou non chargées.")
        
        # Onglet Lithologie
        with stats_tabs[1]:
            st.subheader('Statistiques de lithologie')
            
            if st.session_state.lithology_df is not None and st.session_state.column_mapping['lithology'] and st.session_state.column_mapping['lith_from'] and st.session_state.column_mapping['lith_to']:
                lith_col = st.session_state.column_mapping['lithology']
                from_col = st.session_state.column_mapping['lith_from']
                to_col = st.session_state.column_mapping['lith_to']
                
                # Calculer la longueur de chaque intervalle
                st.session_state.lithology_df['interval_length'] = st.session_state.lithology_df[to_col] - st.session_state.lithology_df[from_col]
                
                # Distribution des lithologies
                lith_counts = st.session_state.lithology_df[lith_col].value_counts()
                st.write(f"Nombre de types de lithologie: {len(lith_counts)}")
                
                # Tableau des statistiques de lithologie
                lith_stats = st.session_state.lithology_df.groupby(lith_col)['interval_length'].agg(['count', 'sum', 'mean', 'min', 'max']).reset_index()
                lith_stats.columns = ['Lithologie', 'Nombre', 'Longueur totale (m)', 'Longueur moyenne (m)', 'Minimum (m)', 'Maximum (m)']
                st.dataframe(lith_stats.sort_values('Longueur totale (m)', ascending=False))
                
                # Graphique de la distribution des lithologies
                fig = px.bar(lith_stats, 
                             x='Lithologie', 
                             y='Longueur totale (m)',
                             color='Lithologie',
                             title="Longueur totale par type de lithologie")
                st.plotly_chart(fig, use_container_width=True)
                
                # Graphique circulaire
                fig_pie = px.pie(lith_stats, 
                                 values='Longueur totale (m)', 
                                 names='Lithologie',
                                 title="Proportion des lithologies")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Données de lithologie incomplètes ou non chargées.")
        
        # Onglet Analyses
        with stats_tabs[2]:
            st.subheader('Statistiques des analyses')
            
            if st.session_state.assays_df is not None and st.session_state.column_mapping['assay_value'] and st.session_state.column_mapping['assay_from'] and st.session_state.column_mapping['assay_to']:
                value_col = st.session_state.column_mapping['assay_value']
                from_col = st.session_state.column_mapping['assay_from']
                to_col = st.session_state.column_mapping['assay_to']
                
                # Calculer la longueur de chaque intervalle
                st.session_state.assays_df['interval_length'] = st.session_state.assays_df[to_col] - st.session_state.assays_df[from_col]
                
                # Statistiques descriptives
                assay_stats = st.session_state.assays_df[value_col].describe()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Statistiques descriptives des teneurs:")
                    st.write(f"Moyenne: {assay_stats['mean']:.3f}")
                    st.write(f"Médiane: {assay_stats['50%']:.3f}")
                    st.write(f"Écart-type: {assay_stats['std']:.3f}")
                    st.write(f"Minimum: {assay_stats['min']:.3f}")
                    st.write(f"Maximum: {assay_stats['max']:.3f}")
                
                with col2:
                    st.write("Percentiles:")
                    st.write(f"25%: {assay_stats['25%']:.3f}")
                    st.write(f"75%: {assay_stats['75%']:.3f}")
                    st.write(f"90%: {assay_stats['75%'] + 1.28 * assay_stats['std']:.3f}")
                    st.write(f"95%: {assay_stats['75%'] + 1.645 * assay_stats['std']:.3f}")
                    st.write(f"99%: {assay_stats['75%'] + 2.326 * assay_stats['std']:.3f}")
                
                # Histogramme des teneurs
                fig = px.histogram(st.session_state.assays_df, 
                                   x=value_col,
                                   title=f"Distribution des teneurs ({value_col})",
                                   labels={value_col: f'Teneur'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Boxplot des teneurs par forage
                if st.session_state.column_mapping['hole_id'] in st.session_state.assays_df.columns:
                    hole_id_col = st.session_state.column_mapping['hole_id']
                    
                    # Nombre maximum de forages à afficher
                    max_holes = 15
                    top_holes = st.session_state.assays_df.groupby(hole_id_col)[value_col].mean().sort_values(ascending=False).head(max_holes).index.tolist()
                    
                    filtered_df = st.session_state.assays_df[st.session_state.assays_df[hole_id_col].isin(top_holes)]
                    
                    fig_box = px.box(filtered_df, 
                                     x=hole_id_col, 
                                     y=value_col,
                                     title=f"Distribution des teneurs par forage (Top {len(top_holes)})")
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Longueur cumulée par classe de teneur
                    threshold = st.slider("Seuil de teneur pour l'analyse", 
                                        min_value=float(st.session_state.assays_df[value_col].min()),
                                        max_value=float(st.session_state.assays_df[value_col].max()),
                                        value=float(st.session_state.assays_df[value_col].mean()))
                    
                    above_threshold = st.session_state.assays_df[st.session_state.assays_df[value_col] >= threshold]
                    
                    st.write(f"Intervalles avec teneur ≥ {threshold}:")
                    st.write(f"Nombre d'intervalles: {len(above_threshold)}")
                    st.write(f"Longueur totale: {above_threshold['interval_length'].sum():.2f} m")
                    st.write(f"Teneur moyenne pondérée: {(above_threshold[value_col] * above_threshold['interval_length']).sum() / above_threshold['interval_length'].sum():.3f}")
            else:
                st.info("Données d'analyses incomplètes ou non chargées.")

# Page de visualisation 3D
elif page == 'Visualisation 3D':
    st.header('Visualisation 3D des forages')
    
    # Vérifier si les données nécessaires ont été chargées
    if st.session_state.collars_df is None or st.session_state.survey_df is None:
        st.warning("Les données de collars et de survey sont nécessaires pour la visualisation 3D. Veuillez les charger d'abord.")
    else:
        # Vérifier si les colonnes nécessaires ont été spécifiées
        required_cols = ['hole_id', 'x', 'y', 'z', 'azimuth', 'dip', 'depth']
        missing_cols = [col for col in required_cols if st.session_state.column_mapping[col] is None or st.session_state.column_mapping[col] == '']
        
        if missing_cols:
            st.warning(f"Certaines colonnes requises n'ont pas été spécifiées: {', '.join(missing_cols)}. Veuillez les définir dans l'onglet 'Chargement des données'.")
        else:
            # Options pour la visualisation
            st.subheader("Options de visualisation")
            
            # Sélection des forages à afficher
            hole_id_col = st.session_state.column_mapping['hole_id']
            all_holes = sorted(st.session_state.collars_df[hole_id_col].unique())
            
            selected_holes = st.multiselect("Sélectionner les forages à afficher", all_holes, default=all_holes[:min(5, len(all_holes))])
            
            # Options additionnelles
            col1, col2 = st.columns(2)
            with col1:
                show_lithology = st.checkbox("Afficher la lithologie", value=True if st.session_state.lithology_df is not None else False)
            with col2:
                show_assays = st.checkbox("Afficher les teneurs", value=True if st.session_state.assays_df is not None else False)
            
            # Filtrer les données selon les forages sélectionnés
            if selected_holes:
                filtered_collars = st.session_state.collars_df[st.session_state.collars_df[hole_id_col].isin(selected_holes)]
                filtered_survey = st.session_state.survey_df[st.session_state.survey_df[hole_id_col].isin(selected_holes)]
                
                # Filtrer lithology et assays si nécessaire
                filtered_lithology = None
                if show_lithology and st.session_state.lithology_df is not None:
                    filtered_lithology = st.session_state.lithology_df[st.session_state.lithology_df[hole_id_col].isin(selected_holes)]
                
                filtered_assays = None
                if show_assays and st.session_state.assays_df is not None:
                    filtered_assays = st.session_state.assays_df[st.session_state.assays_df[hole_id_col].isin(selected_holes)]
                
                # Créer la visualisation 3D
                fig_3d = create_drillhole_3d_plot(
                    filtered_collars, filtered_survey, filtered_lithology, filtered_assays,
                    hole_id_col=hole_id_col,
                    x_col=st.session_state.column_mapping['x'],
                    y_col=st.session_state.column_mapping['y'],
                    z_col=st.session_state.column_mapping['z'],
                    azimuth_col=st.session_state.column_mapping['azimuth'],
                    dip_col=st.session_state.column_mapping['dip'],
                    depth_col=st.session_state.column_mapping['depth'],
                    lith_from_col=st.session_state.column_mapping['lith_from'],
                    lith_to_col=st.session_state.column_mapping['lith_to'],
                    lith_col=st.session_state.column_mapping['lithology'],
                    assay_from_col=st.session_state.column_mapping['assay_from'],
                    assay_to_col=st.session_state.column_mapping['assay_to'],
                    assay_value_col=st.session_state.column_mapping['assay_value']
                )
                
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.error("Impossible de créer la visualisation 3D avec les données fournies.")
            else:
                st.info("Veuillez sélectionner au moins un forage à afficher.")