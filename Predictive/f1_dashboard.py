# f1_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes
from main_script import OpenF1DataPipeline, DriverPerformanceAnalyzer

# Set page configuration
st.set_page_config(
    page_title="F1 Driver Performance Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF1801;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFFFFF;
        background-color: #15151E;
        padding: 10px;
        border-radius: 5px;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF1801;
        margin-bottom: 10px;
    }
    .driver-card {
        background-color: #2D2D44;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        transition: transform 0.2s;
    }
    .driver-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255, 24, 1, 0.2);
    }
    .team-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for caching
if 'scores_data' not in st.session_state:
    st.session_state.scores_data = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

# Driver name and team mapping
DRIVER_MAPPING = {
    1: {"name": "Max Verstappen", "team": "Red Bull", "color": "#3671C6"},
    44: {"name": "Lewis Hamilton", "team": "Mercedes", "color": "#6CD3BF"},
    16: {"name": "Charles Leclerc", "team": "Ferrari", "color": "#F91536"},
    4: {"name": "Lando Norris", "team": "McLaren", "color": "#F58020"},
    55: {"name": "Carlos Sainz", "team": "Ferrari", "color": "#F91536"},
    11: {"name": "Sergio Perez", "team": "Red Bull", "color": "#3671C6"},
    63: {"name": "George Russell", "team": "Mercedes", "color": "#6CD3BF"},
    81: {"name": "Oscar Piastri", "team": "McLaren", "color": "#F58020"},
    14: {"name": "Fernando Alonso", "team": "Aston Martin", "color": "#229971"},
    18: {"name": "Lance Stroll", "team": "Aston Martin", "color": "#229971"},
    22: {"name": "Yuki Tsunoda", "team": "RB", "color": "#6692FF"},
    3: {"name": "Daniel Ricciardo", "team": "RB", "color": "#6692FF"},
    27: {"name": "Nico Hulkenberg", "team": "Haas", "color": "#B6BABD"},
    20: {"name": "Kevin Magnussen", "team": "Haas", "color": "#B6BABD"},
    23: {"name": "Alexander Albon", "team": "Williams", "color": "#64C4FF"},
    2: {"name": "Logan Sargeant", "team": "Williams", "color": "#64C4FF"},
    24: {"name": "Zhou Guanyu", "team": "Sauber", "color": "#C92D4B"},
    77: {"name": "Valtteri Bottas", "team": "Sauber", "color": "#C92D4B"},
    31: {"name": "Esteban Ocon", "team": "Alpine", "color": "#FF87BC"},
    10: {"name": "Pierre Gasly", "team": "Alpine", "color": "#FF87BC"},
    50: {"name": "Oliver Bearman", "team": "Haas", "color": "#B6BABD"}
}

TEAM_COLORS = {
    "Red Bull": "#3671C6",
    "Mercedes": "#6CD3BF",
    "Ferrari": "#F91536",
    "McLaren": "#F58020",
    "Aston Martin": "#229971",
    "RB": "#6692FF",
    "Haas": "#B6BABD",
    "Williams": "#64C4FF",
    "Sauber": "#C92D4B",
    "Alpine": "#FF87BC"
}

def get_driver_info(driver_num):
    """Get driver information from mapping"""
    return DRIVER_MAPPING.get(driver_num, {
        "name": f"Driver {driver_num}",
        "team": "Unknown",
        "color": "#666666"
    })

def load_data():
    """Load and process F1 data"""
    if st.session_state.scores_data is None:
        with st.spinner("🚀 Loading F1 data... This may take a minute..."):
            try:
                pipeline = OpenF1DataPipeline(cache_dir="./f1_data_cache")
                analyzer = DriverPerformanceAnalyzer(pipeline)
                scores = analyzer.calculate_composite_score(2024)
                
                if scores:
                    # Add driver info to scores
                    for driver_num, score_data in scores.items():
                        driver_info = get_driver_info(driver_num)
                        score_data['driver_name'] = driver_info['name']
                        score_data['team'] = driver_info['team']
                        score_data['team_color'] = driver_info['color']
                    
                    st.session_state.scores_data = scores
                    st.session_state.pipeline = pipeline
                    st.session_state.analyzer = analyzer
                    st.success("✅ Data loaded successfully!")
                else:
                    st.error("Failed to load data. Please check your connection.")
                    
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    return st.session_state.scores_data

def create_composite_score_chart(scores_data):
    """Create interactive composite score chart"""
    # Prepare data
    drivers_data = []
    for driver_num, scores in scores_data.items():
        drivers_data.append({
            'Driver': scores.get('driver_name', f'Driver {driver_num}'),
            'Team': scores.get('team', 'Unknown'),
            'Composite Score': scores.get('composite_score', 0),
            'Qualifying': scores.get('quali_score', 0),
            'Race Pace': scores.get('pace_score', 0),
            'Consistency': scores.get('consistency_score', 0),
            'Racecraft': scores.get('racecraft_score', 0),
            'Reliability': scores.get('reliability_score', 0),
            'Team Color': scores.get('team_color', '#666666')
        })
    
    df = pd.DataFrame(drivers_data)
    df = df.sort_values('Composite Score', ascending=False)
    
    # Create figure
    fig = px.bar(df, x='Driver', y='Composite Score',
                 color='Team',
                 color_discrete_map=TEAM_COLORS,
                 title='F1 2024 Season - Driver Composite Scores',
                 hover_data=['Qualifying', 'Race Pace', 'Consistency', 'Racecraft', 'Reliability'],
                 height=600)
    
    fig.update_layout(
        xaxis_title="Driver",
        yaxis_title="Composite Score (0-100)",
        yaxis_range=[0, 100],
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_radar_chart(driver_scores, driver_name):
    """Create radar chart for individual driver"""
    categories = ['Qualifying', 'Race Pace', 'Consistency', 'Racecraft', 'Reliability']
    
    values = [
        driver_scores.get('quali_score', 0),
        driver_scores.get('pace_score', 0),
        driver_scores.get('consistency_score', 0),
        driver_scores.get('racecraft_score', 0),
        driver_scores.get('reliability_score', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=driver_name,
        line_color=driver_scores.get('team_color', '#3671C6')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title=f"{driver_name} - Performance Profile",
        height=500
    )
    
    return fig

def create_component_comparison_chart(scores_data, selected_drivers):
    """Create component comparison chart for selected drivers"""
    categories = ['Qualifying', 'Race Pace', 'Consistency', 'Racecraft', 'Reliability']
    
    fig = go.Figure()
    
    for driver_name in selected_drivers:
        # Find driver data
        driver_scores = None
        for driver_num, scores in scores_data.items():
            if scores.get('driver_name') == driver_name:
                driver_scores = scores
                break
        
        if driver_scores:
            values = [
                driver_scores.get('quali_score', 0),
                driver_scores.get('pace_score', 0),
                driver_scores.get('consistency_score', 0),
                driver_scores.get('racecraft_score', 0),
                driver_scores.get('reliability_score', 0)
            ]
            
            fig.add_trace(go.Bar(
                name=driver_name,
                x=categories,
                y=values,
                marker_color=driver_scores.get('team_color')
            ))
    
    fig.update_layout(
        barmode='group',
        title='Component Score Comparison',
        xaxis_title="Components",
        yaxis_title="Score (0-100)",
        yaxis_range=[0, 100],
        height=500
    )
    
    return fig

def create_team_performance_chart(scores_data):
    """Create team performance analysis chart"""
    # Calculate team averages
    team_stats = {}
    
    for driver_num, scores in scores_data.items():
        team = scores.get('team', 'Unknown')
        if team not in team_stats:
            team_stats[team] = {
                'drivers': [],
                'scores': [],
                'colors': []
            }
        
        team_stats[team]['drivers'].append(scores.get('driver_name', f'Driver {driver_num}'))
        team_stats[team]['scores'].append(scores.get('composite_score', 0))
        team_stats[team]['colors'].append(scores.get('team_color', '#666666'))
    
    # Prepare data for plotting
    teams = []
    avg_scores = []
    max_scores = []
    min_scores = []
    team_colors = []
    
    for team, stats in team_stats.items():
        if stats['scores']:
            teams.append(team)
            avg_scores.append(np.mean(stats['scores']))
            max_scores.append(max(stats['scores']))
            min_scores.append(min(stats['scores']))
            team_colors.append(stats['colors'][0])
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average Score',
        x=teams,
        y=avg_scores,
        marker_color=team_colors,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[max_scores[i] - avg_scores[i] for i in range(len(teams))],
            arrayminus=[avg_scores[i] - min_scores[i] for i in range(len(teams))]
        )
    ))
    
    fig.update_layout(
        title='Team Performance Analysis',
        xaxis_title="Team",
        yaxis_title="Composite Score (0-100)",
        yaxis_range=[0, 100],
        height=500,
        showlegend=False
    )
    
    return fig

def create_score_distribution_chart(scores_data):
    """Create score distribution histogram"""
    scores = [s.get('composite_score', 0) for s in scores_data.values()]
    
    fig = px.histogram(
        x=scores,
        nbins=20,
        title='Composite Score Distribution',
        labels={'x': 'Composite Score', 'y': 'Number of Drivers'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False
    )
    
    # Add mean line
    mean_score = np.mean(scores)
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_score:.1f}")
    
    return fig

def main():
    """Main dashboard function"""
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏎️ F1 Dashboard Settings")
        
        st.markdown("### Season Selection")
        selected_year = st.selectbox(
            "Select Season",
            options=[2024, 2023, 2022],
            index=0
        )
        
        st.markdown("### Analysis Options")
        show_components = st.checkbox("Show Component Breakdown", value=True)
        show_teams = st.checkbox("Show Team Analysis", value=True)
        show_distribution = st.checkbox("Show Score Distribution", value=True)
        
        st.markdown("---")
        
        st.markdown("### Driver Selection")
        scores_data = load_data()
        
        if scores_data:
            driver_names = sorted([s.get('driver_name') for s in scores_data.values()])
            selected_drivers = st.multiselect(
                "Select drivers for comparison",
                options=driver_names,
                default=driver_names[:3] if len(driver_names) >= 3 else driver_names
            )
        
        st.markdown("---")
        
        st.markdown("### About")
        st.info("""
        **F1 Driver Performance Dashboard**
        
        This dashboard analyzes F1 driver performance using:
        - Qualifying (25%)
        - Race Pace (30%)
        - Consistency (20%)
        - Racecraft (15%)
        - Reliability (10%)
        
        Data source: OpenF1 API
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">🏎️ F1 DRIVER PERFORMANCE DASHBOARD</h1>', unsafe_allow_html=True)
    
    if scores_data is None:
        st.warning("Please wait while data is loading...")
        if st.button("Load Data Now"):
            load_data()
            st.rerun()
        return
    
    # Top metrics row
    st.markdown("## 📊 Season Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_drivers = len(scores_data)
        st.metric("Total Drivers", total_drivers)
    
    with col2:
        all_scores = [s.get('composite_score', 0) for s in scores_data.values()]
        avg_score = np.mean(all_scores)
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col3:
        top_driver = max(scores_data.items(), key=lambda x: x[1].get('composite_score', 0))
        top_name = top_driver[1].get('driver_name', 'Unknown')
        top_score = top_driver[1].get('composite_score', 0)
        st.metric("Top Performer", top_name, f"{top_score:.1f}")
    
    with col4:
        sorted_scores = sorted([s.get('composite_score', 0) for s in scores_data.values()])
        median_score = np.median(sorted_scores)
        st.metric("Median Score", f"{median_score:.1f}")
    
    # Main visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overall Rankings", "👤 Driver Analysis", "🏎️ Team Analysis", "📋 Detailed Data"])
    
    with tab1:
        st.markdown("### Driver Composite Scores")
        fig1 = create_composite_score_chart(scores_data)
        st.plotly_chart(fig1, use_container_width=True)
        
        if show_distribution:
            st.markdown("### Score Distribution")
            fig_dist = create_score_distribution_chart(scores_data)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        st.markdown("### Individual Driver Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_driver = st.selectbox(
                "Select Driver",
                options=[s.get('driver_name') for s in scores_data.values()],
                index=0
            )
            
            # Find selected driver data
            driver_scores = None
            for driver_num, scores in scores_data.items():
                if scores.get('driver_name') == selected_driver:
                    driver_scores = scores
                    break
            
            if driver_scores:
                # Create radar chart
                radar_fig = create_radar_chart(driver_scores, selected_driver)
                st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            if driver_scores:
                st.markdown(f"### {selected_driver}")
                
                # Driver info card
                team = driver_scores.get('team', 'Unknown')
                team_color = driver_scores.get('team_color', '#666666')
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Driver Information</h4>
                    <p><strong>Team:</strong> <span style="color:{team_color}">{team}</span></p>
                    <p><strong>Composite Score:</strong> {driver_scores.get('composite_score', 0):.1f}/100</p>
                    <hr>
                    <h5>Component Scores:</h5>
                    <p>• Qualifying: {driver_scores.get('quali_score', 0):.1f}</p>
                    <p>• Race Pace: {driver_scores.get('pace_score', 0):.1f}</p>
                    <p>• Consistency: {driver_scores.get('consistency_score', 0):.1f}</p>
                    <p>• Racecraft: {driver_scores.get('racecraft_score', 0):.1f}</p>
                    <p>• Reliability: {driver_scores.get('reliability_score', 0):.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Component comparison
        if show_components and len(selected_drivers) > 1:
            st.markdown("### Driver Comparison")
            comp_fig = create_component_comparison_chart(scores_data, selected_drivers)
            st.plotly_chart(comp_fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Team Performance Analysis")
        
        if show_teams:
            team_fig = create_team_performance_chart(scores_data)
            st.plotly_chart(team_fig, use_container_width=True)
        
        # Team ranking table
        st.markdown("### Team Rankings")
        
        # Calculate team averages
        team_averages = {}
        for driver_num, scores in scores_data.items():
            team = scores.get('team', 'Unknown')
            if team not in team_averages:
                team_averages[team] = []
            team_averages[team].append(scores.get('composite_score', 0))
        
        # Create team ranking DataFrame
        team_data = []
        for team, scores in team_averages.items():
            team_data.append({
                'Team': team,
                'Average Score': np.mean(scores),
                'Highest Score': max(scores),
                'Lowest Score': min(scores),
                'Driver Count': len(scores)
            })
        
        team_df = pd.DataFrame(team_data)
        team_df = team_df.sort_values('Average Score', ascending=False)
        team_df['Rank'] = range(1, len(team_df) + 1)
        
        # Display team table
        st.dataframe(
            team_df[['Rank', 'Team', 'Average Score', 'Highest Score', 'Lowest Score', 'Driver Count']],
            use_container_width=True,
            hide_index=True
        )
    
    with tab4:
        st.markdown("### Complete Driver Data")
        
        # Prepare data for display
        display_data = []
        for driver_num, scores in scores_data.items():
            display_data.append({
                'Rank': None,  # Will be filled after sorting
                'Driver': scores.get('driver_name', f'Driver {driver_num}'),
                'Team': scores.get('team', 'Unknown'),
                'Composite Score': scores.get('composite_score', 0),
                'Qualifying': scores.get('quali_score', 0),
                'Race Pace': scores.get('pace_score', 0),
                'Consistency': scores.get('consistency_score', 0),
                'Racecraft': scores.get('racecraft_score', 0),
                'Reliability': scores.get('reliability_score', 0)
            })
        
        df_display = pd.DataFrame(display_data)
        df_display = df_display.sort_values('Composite Score', ascending=False)
        df_display['Rank'] = range(1, len(df_display) + 1)
        
        # Display table with filters
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Filters")
            min_score = st.slider("Minimum Score", 0, 100, 0)
            selected_teams = st.multiselect(
                "Filter by Team",
                options=df_display['Team'].unique(),
                default=df_display['Team'].unique()
            )
        
        # Apply filters
        filtered_df = df_display[
            (df_display['Composite Score'] >= min_score) &
            (df_display['Team'].isin(selected_teams))
        ]
        
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Rank': st.column_config.NumberColumn(format="%d"),
                'Composite Score': st.column_config.NumberColumn(format="%.1f"),
                'Qualifying': st.column_config.NumberColumn(format="%.1f"),
                'Race Pace': st.column_config.NumberColumn(format="%.1f"),
                'Consistency': st.column_config.NumberColumn(format="%.1f"),
                'Racecraft': st.column_config.NumberColumn(format="%.1f"),
                'Reliability': st.column_config.NumberColumn(format="%.1f")
            }
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="f1_driver_performance.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>F1 Driver Performance Dashboard • Data Source: OpenF1 API • Created with Streamlit</p>
        <p>Scoring Methodology: Qualifying (25%) • Race Pace (30%) • Consistency (20%) • Racecraft (15%) • Reliability (10%)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()