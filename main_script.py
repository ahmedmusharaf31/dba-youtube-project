# main_script.py - COMPLETE FIXED VERSION
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA PIPELINE CLASSES
# ============================================================================

class OpenF1DataPipeline:
    """A comprehensive pipeline for fetching and processing F1 data from OpenF1 API"""
    
    BASE_URL = "https://api.openf1.org/v1"
    
    def __init__(self, cache_dir="./f1_cache"):
        self.session = requests.Session()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _make_request(self, endpoint, params=None, use_cache=True):
        """Make API request with caching support"""
        cache_key = f"{endpoint}_{hash(frozenset(params.items()) if params else '')}.json"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return pd.DataFrame(json.load(f))
        
        try:
            response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            return pd.DataFrame(data) if data else pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return pd.DataFrame()
    
    # All your existing methods remain the same...
    def get_seasons(self, year=None):
        """Get season information"""
        params = {"year": year} if year else {}
        return self._make_request("seasons", params)
    
    def get_meetings(self, year, country_name=None):
        """Get all races (meetings) for a season"""
        params = {"year": year}
        if country_name:
            params["country_name"] = country_name
        return self._make_request("meetings", params)
    
    def get_sessions(self, meeting_key, session_type=None):
        """Get sessions for a specific race meeting"""
        params = {"meeting_key": meeting_key}
        if session_type:
            params["session_type"] = session_type
        return self._make_request("sessions", params)
    
    def get_drivers(self, session_key=None, driver_number=None):
        """Get driver information"""
        params = {}
        if session_key:
            params["session_key"] = session_key
        if driver_number:
            params["driver_number"] = driver_number
        return self._make_request("drivers", params)
    
    def get_laps(self, session_key, driver_number=None):
        """Get lap times for a session"""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        return self._make_request("laps", params)
    
    def get_intervals(self, session_key):
        """Get interval data (gap to car ahead)"""
        params = {"session_key": session_key}
        return self._make_request("intervals", params)
    
    def get_position(self, session_key):
        """Get position data"""
        params = {"session_key": session_key}
        return self._make_request("position", params)
    
    def get_race_control(self, session_key):
        """Get race control messages (flags, safety car, etc.)"""
        params = {"session_key": session_key}
        return self._make_request("race_control", params)
    
    def get_team_radios(self, session_key):
        """Get team radio messages"""
        params = {"session_key": session_key}
        return self._make_request("team_radios", params)
    
    def get_car_data(self, session_key, driver_number=None):
        """Get telemetry data (speed, throttle, brake, etc.)"""
        params = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        return self._make_request("car_data", params)
    
    def get_weather(self, session_key):
        """Get weather data"""
        params = {"session_key": session_key}
        return self._make_request("weather", params)
    
    def get_pit(self, session_key):
        """Get pit stop data"""
        params = {"session_key": session_key}
        return self._make_request("pit", params)
    
    def get_stints(self, session_key):
        """Get stint data"""
        params = {"session_key": session_key}
        return self._make_request("stints", params)
    
    def get_qualifying_results(self, session_key):
        """Get qualifying results"""
        # OpenF1 doesn't have a direct qualifying endpoint, but we can get from laps
        laps = self.get_laps(session_key)
        if laps.empty:
            return pd.DataFrame()
        
        # Filter for qualifying laps (lap_type = 'qualifying' or Q sessions)
        quali_laps = laps[laps['lap_duration'].notna()].copy()
        
        # Group by driver and get best lap
        quali_results = quali_laps.groupby('driver_number').agg({
            'lap_duration': 'min',
            'lap_number': 'first'
        }).reset_index()
        
        # Sort by fastest lap
        quali_results = quali_results.sort_values('lap_duration')
        quali_results['position'] = range(1, len(quali_results) + 1)
        
        return quali_results
    
    def get_race_results(self, session_key):
        """Get race results"""
        # Get position data at the end of the race
        position_data = self.get_position(session_key)
        if position_data.empty:
            return pd.DataFrame()
        
        # Get the last position for each driver
        last_positions = position_data.sort_values('date').groupby('driver_number').last().reset_index()
        
        # Merge with driver info
        drivers = self.get_drivers(session_key)
        if not drivers.empty:
            last_positions = last_positions.merge(
                drivers[['driver_number', 'full_name', 'team_name']],
                on='driver_number',
                how='left'
            )
        
        return last_positions.sort_values('position')
    
    def get_complete_weekend_data(self, meeting_key):
        """Get all relevant data for a race weekend"""
        print(f"Fetching data for meeting: {meeting_key}")
        
        # Get sessions for this meeting
        sessions = self.get_sessions(meeting_key)
        if sessions.empty:
            return {}
        
        weekend_data = {}
        
        for _, session in sessions.iterrows():
            session_key = session['session_key']
            session_type = session['session_name']
            
            print(f"  Processing {session_type} (key: {session_key})")
            
            session_data = {
                'session_info': session.to_dict(),
                'drivers': self.get_drivers(session_key),
                'laps': self.get_laps(session_key),
                'position': self.get_position(session_key),
                'intervals': self.get_intervals(session_key),
                'weather': self.get_weather(session_key),
                'pit': self.get_pit(session_key),
                'stints': self.get_stints(session_key),
                'race_control': self.get_race_control(session_key)
            }
            
            weekend_data[session_type] = session_data
            
            # Be nice to the API
            time.sleep(0.5)
        
        return weekend_data

# ============================================================================
# 2. PERFORMANCE ANALYZER CLASS
# ============================================================================

class DriverPerformanceAnalyzer:
    """Analyzes driver performance and calculates composite scores"""
    
    def __init__(self, data_pipeline):
        self.pipeline = data_pipeline
        self.driver_stats = {}
    
    def fetch_season_data(self, year):
        """Fetch all race data for a season"""
        print(f"Fetching data for {year} season...")
        
        season_data = {}
        meetings = self.pipeline.get_meetings(year)
        
        for _, meeting in meetings.iterrows():
            meeting_key = meeting['meeting_key']
            meeting_name = meeting['meeting_name']
            print(f"  Processing {meeting_name}")
            
            weekend_data = self.pipeline.get_complete_weekend_data(meeting_key)
            if weekend_data:
                season_data[meeting_key] = {
                    'meeting_info': meeting.to_dict(),
                    'weekend_data': weekend_data
                }
        
        return season_data
    
    def calculate_qualifying_score(self, season_data):
        """Calculate qualifying performance metrics"""
        print("Calculating qualifying scores...")
        
        quali_scores = {}
        
        for meeting_key, meeting_data in season_data.items():
            weekend_data = meeting_data['weekend_data']
            
            # Look for qualifying sessions
            for session_type, session_data in weekend_data.items():
                if 'qualifying' in session_type.lower() or 'q' in session_type.lower():
                    laps_df = session_data['laps']
                    
                    if laps_df.empty:
                        continue
                    
                    # Get best lap for each driver
                    driver_laps = laps_df[laps_df['lap_duration'].notna()].copy()
                    if driver_laps.empty:
                        continue
                    
                    best_laps = driver_laps.groupby('driver_number').agg({
                        'lap_duration': 'min',
                        'lap_number': 'first'
                    }).reset_index()
                    
                    # Calculate position
                    best_laps = best_laps.sort_values('lap_duration')
                    best_laps['qualifying_position'] = range(1, len(best_laps) + 1)
                    
                    # Store results
                    for _, row in best_laps.iterrows():
                        driver = row['driver_number']
                        if driver not in quali_scores:
                            quali_scores[driver] = []
                        
                        quali_scores[driver].append({
                            'meeting': meeting_key,
                            'position': row['qualifying_position'],
                            'time': row['lap_duration']
                        })
        
        return quali_scores
    
    def calculate_race_pace_score(self, season_data):
        """Calculate race pace metrics (median lap time, normalized)"""
        print("Calculating race pace scores...")
        
        pace_scores = {}
        
        for meeting_key, meeting_data in season_data.items():
            weekend_data = meeting_data['weekend_data']
            
            # Look for race sessions
            for session_type, session_data in weekend_data.items():
                if 'race' in session_type.lower():
                    laps_df = session_data['laps']
                    
                    if laps_df.empty:
                        continue
                    
                    # Filter out problematic laps
                    valid_laps = laps_df[
                        (laps_df['lap_duration'].notna()) &
                        (laps_df['lap_duration'] > 0) &
                        (laps_df['lap_number'] > 1)  # Exclude first lap
                    ].copy()
                    
                    if valid_laps.empty:
                        continue
                    
                    # Get driver teams for normalization
                    drivers_df = session_data['drivers']
                    
                    # Group by driver and calculate median lap time
                    driver_pace = valid_laps.groupby('driver_number').agg({
                        'lap_duration': ['median', 'std', 'count']
                    }).reset_index()
                    
                    driver_pace.columns = ['driver_number', 'median_lap', 'lap_std', 'lap_count']
                    
                    # Merge with team info
                    if not drivers_df.empty:
                        driver_pace = driver_pace.merge(
                            drivers_df[['driver_number', 'team_name']],
                            on='driver_number',
                            how='left'
                        )
                    
                    # Calculate normalized pace within team
                    for team in driver_pace['team_name'].unique():
                        team_drivers = driver_pace[driver_pace['team_name'] == team]
                        if len(team_drivers) > 1:
                            fastest_median = team_drivers['median_lap'].min()
                            for _, row in team_drivers.iterrows():
                                driver = row['driver_number']
                                gap_to_teammate = row['median_lap'] - fastest_median
                                
                                if driver not in pace_scores:
                                    pace_scores[driver] = []
                                
                                pace_scores[driver].append({
                                    'meeting': meeting_key,
                                    'median_lap': row['median_lap'],
                                    'gap_to_teammate': gap_to_teammate,
                                    'consistency': row['lap_std'],
                                    'laps_analyzed': row['lap_count']
                                })
        
        return pace_scores
    
    def calculate_racecraft_score(self, season_data):
        """Calculate positions gained and racecraft metrics"""
        print("Calculating racecraft scores...")
        
        racecraft_scores = {}
        
        for meeting_key, meeting_data in season_data.items():
            weekend_data = meeting_data['weekend_data']
            
            # Get grid positions (from qualifying) and finish positions
            grid_positions = {}
            finish_positions = {}
            
            for session_type, session_data in weekend_data.items():
                # Get qualifying positions
                if 'qualifying' in session_type.lower() or 'q' in session_type.lower():
                    laps_df = session_data['laps']
                    if not laps_df.empty:
                        # Simplified grid position calculation
                        valid_laps = laps_df[laps_df['lap_duration'].notna()]
                        if not valid_laps.empty:
                            best_laps = valid_laps.groupby('driver_number')['lap_duration'].min()
                            sorted_drivers = best_laps.sort_values().index.tolist()
                            for i, driver in enumerate(sorted_drivers):
                                grid_positions[driver] = i + 1
            
            # Get race finish positions
            for session_type, session_data in weekend_data.items():
                if 'race' in session_type.lower():
                    position_df = session_data['position']
                    if not position_df.empty:
                        # Get last recorded position for each driver
                        last_positions = position_df.sort_values('date').groupby('driver_number').last()
                        for driver, row in last_positions.iterrows():
                            finish_positions[driver] = row['position']
            
            # Calculate positions gained
            for driver in set(list(grid_positions.keys()) + list(finish_positions.keys())):
                grid = grid_positions.get(driver)
                finish = finish_positions.get(driver)
                
                if grid is not None and finish is not None:
                    positions_gained = grid - finish  # Positive = gained positions
                    
                    if driver not in racecraft_scores:
                        racecraft_scores[driver] = []
                    
                    racecraft_scores[driver].append({
                        'meeting': meeting_key,
                        'grid_position': grid,
                        'finish_position': finish,
                        'positions_gained': positions_gained
                    })
        
        return racecraft_scores
    
    def calculate_reliability_score(self, season_data):
        """Calculate reliability and DNF metrics"""
        print("Calculating reliability scores...")
        
        reliability_scores = {}
        driver_teams = {}
        
        # First, map drivers to teams
        for meeting_key, meeting_data in season_data.items():
            weekend_data = meeting_data['weekend_data']
            for session_type, session_data in weekend_data.items():
                if not session_data['drivers'].empty:
                    for _, driver_row in session_data['drivers'].iterrows():
                        driver = driver_row['driver_number']
                        driver_teams[driver] = driver_row['team_name']
        
        # Count races and DNFs
        for driver in driver_teams.keys():
            reliability_scores[driver] = {
                'races_started': 0,
                'races_finished': 0,
                'dnf_reasons': []
            }
        
        for meeting_key, meeting_data in season_data.items():
            weekend_data = meeting_data['weekend_data']
            
            for session_type, session_data in weekend_data.items():
                if 'race' in session_type.lower():
                    # Get all drivers in the race
                    drivers_df = session_data['drivers']
                    if drivers_df.empty:
                        continue
                    
                    for driver in drivers_df['driver_number'].unique():
                        if driver in reliability_scores:
                            reliability_scores[driver]['races_started'] += 1
                    
                    # Check DNFs from position data or intervals
                    position_df = session_data['position']
                    if not position_df.empty:
                        # Check if driver was running at the end
                        last_positions = position_df.sort_values('date').groupby('driver_number').last()
                        # This is simplified - in reality you'd check race control for retirements
                        for driver in last_positions.index:
                            if driver in reliability_scores:
                                reliability_scores[driver]['races_finished'] += 1
        
        return reliability_scores
    
    def normalize_scores(self, scores_dict, reverse=False):
        """Normalize scores to 0-100 scale"""
        if not scores_dict:
            return {}
        
        values = list(scores_dict.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 50 for k in scores_dict.keys()}
        
        normalized = {}
        for driver, score in scores_dict.items():
            if reverse:  # For metrics where lower is better (like lap times)
                norm_score = 100 * (max_val - score) / (max_val - min_val)
            else:
                norm_score = 100 * (score - min_val) / (max_val - min_val)
            normalized[driver] = norm_score
        
        return normalized
    
    def calculate_composite_score(self, year):
        """Calculate the complete composite score for all drivers in a season"""
        print(f"\n{'='*50}")
        print(f"Calculating composite scores for {year}")
        print(f"{'='*50}")
        
        # Fetch season data
        season_data = self.fetch_season_data(year)
        
        if not season_data:
            print(f"No data found for {year}")
            return {}
        
        # Calculate all component scores
        quali_scores = self.calculate_qualifying_score(season_data)
        pace_scores = self.calculate_race_pace_score(season_data)
        racecraft_scores = self.calculate_racecraft_score(season_data)
        reliability_scores = self.calculate_reliability_score(season_data)
        
        # Aggregate scores per driver
        driver_summary = {}
        
        # Process qualifying scores
        for driver, races in quali_scores.items():
            if driver not in driver_summary:
                driver_summary[driver] = {
                    'driver_number': driver,
                    'qualifying_positions': [],
                    'qualifying_times': []
                }
            
            for race in races:
                driver_summary[driver]['qualifying_positions'].append(race['position'])
                driver_summary[driver]['qualifying_times'].append(race['time'])
        
        # Process pace scores
        for driver, races in pace_scores.items():
            if driver not in driver_summary:
                driver_summary[driver] = {'driver_number': driver}
            
            driver_summary[driver].setdefault('median_laps', [])
            driver_summary[driver].setdefault('pace_gaps', [])
            driver_summary[driver].setdefault('consistency_values', [])
            
            for race in races:
                driver_summary[driver]['median_laps'].append(race['median_lap'])
                driver_summary[driver]['pace_gaps'].append(race['gap_to_teammate'])
                driver_summary[driver]['consistency_values'].append(race['consistency'])
        
        # Process racecraft scores
        for driver, races in racecraft_scores.items():
            if driver not in driver_summary:
                driver_summary[driver] = {'driver_number': driver}
            
            driver_summary[driver].setdefault('positions_gained', [])
            
            for race in races:
                driver_summary[driver]['positions_gained'].append(race['positions_gained'])
        
        # Process reliability scores
        for driver, stats in reliability_scores.items():
            if driver not in driver_summary:
                driver_summary[driver] = {'driver_number': driver}
            
            driver_summary[driver]['reliability'] = {
                'races_started': stats['races_started'],
                'races_finished': stats['races_finished'],
                'finish_rate': stats['races_finished'] / stats['races_started'] if stats['races_started'] > 0 else 0
            }
        
        # Calculate aggregated metrics
        composite_scores = {}
        
        for driver, stats in driver_summary.items():
            # A. Qualifying Score (25%)
            if 'qualifying_positions' in stats and stats['qualifying_positions']:
                avg_quali_position = np.mean(stats['qualifying_positions'])
                quali_score = 100 * (1 - (avg_quali_position - 1) / 19)  # Normalize to 1-20 grid
            else:
                quali_score = 0
            
            # B. Race Pace Score (30%)
            if 'pace_gaps' in stats and stats['pace_gaps']:
                avg_pace_gap = np.mean(stats['pace_gaps'])
                # Convert gap to score (smaller gap = better)
                pace_score = 100 * max(0, 1 - avg_pace_gap / 0.5)  # Assuming 0.5s gap is threshold
            else:
                pace_score = 0
            
            # C. Consistency Score (20%)
            if 'consistency_values' in stats and stats['consistency_values']:
                avg_consistency = np.mean(stats['consistency_values'])
                consistency_score = 100 * max(0, 1 - avg_consistency / 0.5)  # Assuming 0.5s std is threshold
            else:
                consistency_score = 0
            
            # D. Racecraft Score (15%)
            if 'positions_gained' in stats and stats['positions_gained']:
                avg_positions_gained = np.mean(stats['positions_gained'])
                racecraft_score = 100 * (avg_positions_gained + 10) / 20  # Normalize -10 to +10 range to 0-100
            else:
                racecraft_score = 0
            
            # E. Reliability Score (10%)
            if 'reliability' in stats:
                finish_rate = stats['reliability']['finish_rate']
                reliability_score = finish_rate * 100
            else:
                reliability_score = 0
            
            # Composite Score with weights
            composite_score = (
                0.25 * quali_score +
                0.30 * pace_score +
                0.20 * consistency_score +
                0.15 * racecraft_score +
                0.10 * reliability_score
            )
            
            composite_scores[driver] = {
                'driver_number': driver,
                'quali_score': round(quali_score, 1),
                'pace_score': round(pace_score, 1),
                'consistency_score': round(consistency_score, 1),
                'racecraft_score': round(racecraft_score, 1),
                'reliability_score': round(reliability_score, 1),
                'composite_score': round(composite_score, 1),
                'component_scores': {
                    'avg_quali_position': round(avg_quali_position, 2) if 'qualifying_positions' in stats else None,
                    'avg_pace_gap': round(avg_pace_gap, 3) if 'pace_gaps' in stats else None,
                    'avg_consistency': round(avg_consistency, 3) if 'consistency_values' in stats else None,
                    'avg_positions_gained': round(avg_positions_gained, 2) if 'positions_gained' in stats else None,
                    'finish_rate': round(finish_rate, 3) if 'reliability' in stats else None
                }
            }
        
        return composite_scores

# ============================================================================
# 3. VISUALIZATION CLASS
# ============================================================================

class PerformanceVisualizer:
    """Visualizes driver performance data with driver names"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Driver name mapping for cleaner display
        self.name_shortcuts = {
            "Max Verstappen": "Verstappen",
            "Lewis Hamilton": "Hamilton", 
            "Charles Leclerc": "Leclerc",
            "Lando Norris": "Norris",
            "Carlos Sainz": "Sainz",
            "Sergio Perez": "Perez",
            "George Russell": "Russell",
            "Oscar Piastri": "Piastri",
            "Fernando Alonso": "Alonso",
            "Lance Stroll": "Stroll",
            "Yuki Tsunoda": "Tsunoda",
            "Daniel Ricciardo": "Ricciardo",
            "Nico Hulkenberg": "Hulkenberg",
            "Kevin Magnussen": "Magnussen",
            "Alexander Albon": "Albon",
            "Logan Sargeant": "Sargeant",
            "Zhou Guanyu": "Zhou",
            "Valtteri Bottas": "Bottas",
            "Esteban Ocon": "Ocon",
            "Pierre Gasly": "Gasly"
        }
    
    def _get_display_name(self, driver_name):
        """Get shortened display name for graphs"""
        if driver_name in self.name_shortcuts:
            return self.name_shortcuts[driver_name]
        # If driver name is long, take first name
        if len(driver_name) > 12:
            return driver_name.split()[0]
        return driver_name
    
    def plot_composite_scores(self, composite_scores, year):
        """Create a bar chart of composite scores with driver names"""
        if not composite_scores:
            print("No data to visualize")
            return
        
        # Convert to list and sort
        drivers_data = []
        for driver_num, scores_dict in composite_scores.items():
            driver_name = scores_dict.get('driver_name', f"Driver {driver_num}")
            drivers_data.append({
                'driver_number': driver_num,
                'driver_name': driver_name,
                'display_name': self._get_display_name(driver_name),
                'composite_score': scores_dict.get('composite_score', 0),
                'quali_score': scores_dict.get('quali_score', 0),
                'pace_score': scores_dict.get('pace_score', 0),
                'consistency_score': scores_dict.get('consistency_score', 0),
                'racecraft_score': scores_dict.get('racecraft_score', 0),
                'reliability_score': scores_dict.get('reliability_score', 0)
            })
        
        # Sort by composite score
        drivers_data.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Take top 15 drivers for better visualization
        top_n = min(15, len(drivers_data))
        top_drivers = drivers_data[:top_n]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # Bar chart of composite scores WITH NAMES
        driver_names = [d['display_name'] for d in top_drivers]
        scores = [d['composite_score'] for d in top_drivers]
        
        # Create color gradient based on score
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_drivers)))
        
        bars = ax1.bar(range(top_n), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title(f'F1 {year} Season - Top {top_n} Driver Performance Scores', 
                     fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Composite Score (0-100)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.set_xticks(range(top_n))
        ax1.set_xticklabels(driver_names, rotation=45, ha='right', fontsize=10)
        
        # Add value labels on bars
        for bar, score, driver in zip(bars, scores, top_drivers):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9)
            
            # Add driver number below name for reference
            driver_num = driver['driver_number']
            if isinstance(driver_num, (int, float)):
                ax1.text(bar.get_x() + bar.get_width()/2., -5,
                        f'#{int(driver_num)}', ha='center', va='top',
                        fontsize=8, alpha=0.7, color='gray')
        
        # Add grid for better readability
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Stacked bar chart of component scores WITH NAMES
        component_data = []
        for driver in top_drivers:
            component_data.append([
                driver['quali_score'],
                driver['pace_score'], 
                driver['consistency_score'],
                driver['racecraft_score'],
                driver['reliability_score']
            ])
        
        component_df = pd.DataFrame(component_data, index=driver_names,
                                  columns=['Qualifying', 'Race Pace', 'Consistency', 
                                           'Racecraft', 'Reliability'])
        
        component_df.plot(kind='bar', stacked=True, ax=ax2, 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_title('Component Score Breakdown', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Component Scores', fontsize=12)
        ax2.set_xlabel('Driver', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.set_xticklabels(driver_names, rotation=45, ha='right', fontsize=10)
        ax2.legend(title='Performance Components', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add component percentages annotation
        total_scores = component_df.sum(axis=1)
        for i, (idx, row) in enumerate(component_df.iterrows()):
            y_offset = 0
            for component, value in row.items():
                if value > 5:  # Only label if significant
                    percentage = (value / total_scores[idx]) * 100
                    ax2.text(i, y_offset + value/2, f'{value:.0f}',
                            ha='center', va='center', fontsize=8,
                            fontweight='bold', color='white')
                y_offset += value
        
        plt.tight_layout()
        plt.show()
        
        # Also save the figure
        plt.savefig(f'f1_driver_scores_{year}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Chart saved as 'f1_driver_scores_{year}.png'")
    
    def plot_radar_chart(self, driver_scores, title="Driver Performance"):
        """Create a radar chart for a specific driver with name in title"""
        if not driver_scores:
            return
        
        # Extract driver name from title or scores
        driver_name = "Unknown Driver"
        if 'driver_name' in driver_scores:
            driver_name = driver_scores['driver_name']
        elif ":" in title:
            # Extract from title like "Top Performer: Max Verstappen"
            driver_name = title.split(":")[-1].strip()
        
        categories = ['Qualifying', 'Race Pace', 'Consistency', 'Racecraft', 'Reliability']
        
        # Get scores with defaults
        values = [
            driver_scores.get('quali_score', 0),
            driver_scores.get('pace_score', 0),
            driver_scores.get('consistency_score', 0),
            driver_scores.get('racecraft_score', 0),
            driver_scores.get('reliability_score', 0)
        ]
        
        # Calculate composite score
        composite_score = driver_scores.get('composite_score', 
                                          sum(values) / len(values) if values else 0)
        
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values += values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], categories, size=13, fontweight='bold')
        
        # Draw ylabels
        ax.set_rlabel_position(30)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], 
                  color="gray", size=11)
        plt.ylim(0, 100)
        
        # Plot data with gradient fill
        ax.plot(angles, values, linewidth=3, linestyle='solid', 
                color='dodgerblue', marker='o', markersize=8)
        ax.fill(angles, values, 'dodgerblue', alpha=0.3)
        
        # Add value labels at each point
        for angle, value, category in zip(angles[:-1], values[:-1], categories):
            x = angle
            y = value + 5  # Offset from point
            ax.text(x, y, f'{value:.1f}', ha='center', va='center', 
                   fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add a title with driver name and composite score
        plt.title(f'{driver_name}\nPerformance Profile\n'
                 f'Composite Score: {composite_score:.1f}/100', 
                 size=18, fontweight='bold', pad=40)
        
        # Add component weights in subtitle
        fig.text(0.5, 0.92, 'Weights: Qualifying(25%) | Race Pace(30%) | Consistency(20%) | Racecraft(15%) | Reliability(10%)', 
                ha='center', fontsize=11, style='italic', alpha=0.7)
        
        # Add driver info box if available
        info_text = ""
        if 'driver_number' in driver_scores:
            info_text += f"Driver #: {driver_scores['driver_number']}\n"
        if 'team' in driver_scores:
            info_text += f"Team: {driver_scores['team']}"
        
        if info_text:
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
        
        plt.tight_layout()
        plt.show()
        
        # Save the radar chart
        safe_name = driver_name.replace(" ", "_").lower()
        plt.savefig(f'radar_chart_{safe_name}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Radar chart saved as 'radar_chart_{safe_name}.png'")
    
    def create_performance_report(self, composite_scores, year):
        """Generate a comprehensive performance report with names"""
        if not composite_scores:
            return "No data available for report generation."
        
        # Sort drivers by composite score
        sorted_drivers = sorted(composite_scores.items(), 
                               key=lambda x: x[1].get('composite_score', 0), 
                               reverse=True)
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"F1 {year} SEASON - DRIVER PERFORMANCE ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("\n")
        
        # Top performers WITH NAMES
        report_lines.append("TOP 10 PERFORMERS")
        report_lines.append("-" * 40)
        
        for i, (driver_num, scores) in enumerate(sorted_drivers[:10], 1):
            driver_name = scores.get('driver_name', f"Driver {driver_num}")
            team = scores.get('team', 'Unknown')
            
            report_lines.append(f"{i}. {driver_name} ({team}): {scores.get('composite_score', 0):.1f} points")
            report_lines.append(f"   Components: Q:{scores.get('quali_score', 0):.1f} | "
                              f"P:{scores.get('pace_score', 0):.1f} | "
                              f"C:{scores.get('consistency_score', 0):.1f} | "
                              f"R:{scores.get('racecraft_score', 0):.1f} | "
                              f"Rel:{scores.get('reliability_score', 0):.1f}")
        
        report_lines.append("\n")
        
        # Detailed analysis
        report_lines.append("COMPONENT ANALYSIS")
        report_lines.append("-" * 40)
        
        # Find best in each category
        categories = {
            'Qualifying': ('quali_score', sorted_drivers, max, 'Highest'),
            'Race Pace': ('pace_score', sorted_drivers, max, 'Highest'),
            'Consistency': ('consistency_score', sorted_drivers, max, 'Highest'),
            'Racecraft': ('racecraft_score', sorted_drivers, max, 'Highest'),
            'Reliability': ('reliability_score', sorted_drivers, max, 'Highest')
        }
        
        for category, (score_key, drivers, func, desc) in categories.items():
            best_driver = func(drivers, key=lambda x: x[1].get(score_key, 0))
            driver_num, scores = best_driver
            driver_name = scores.get('driver_name', f"Driver {driver_num}")
            score_value = scores.get(score_key, 0)
            report_lines.append(f"{category}: {driver_name} ({score_value:.1f})")
        
        report_lines.append("\n")
        
        # Weight justification
        report_lines.append("SCORING WEIGHTS JUSTIFICATION")
        report_lines.append("-" * 40)
        report_lines.append("1. Race Pace (30%): Most direct measure of raw speed and car control")
        report_lines.append("2. Qualifying (25%): Grid position is crucial for race outcome")
        report_lines.append("3. Consistency (20%): Elite drivers minimize mistakes and lap time variance")
        report_lines.append("4. Racecraft (15%): Overtaking ability and race intelligence")
        report_lines.append("5. Reliability (10%): Must finish races to score points")
        
        # Performance distribution
        report_lines.append("\n")
        report_lines.append("PERFORMANCE DISTRIBUTION")
        report_lines.append("-" * 40)
        
        all_scores = [s[1].get('composite_score', 0) for s in sorted_drivers]
        if all_scores:
            report_lines.append(f"• Number of drivers analyzed: {len(sorted_drivers)}")
            report_lines.append(f"• Average score: {np.mean(all_scores):.1f}")
            report_lines.append(f"• Score range: {min(all_scores):.1f} - {max(all_scores):.1f}")
            report_lines.append(f"• Top score: {max(all_scores):.1f} ({sorted_drivers[0][1].get('driver_name', 'Top driver')})")
            report_lines.append(f"• Bottom score: {min(all_scores):.1f}")
        
        return "\n".join(report_lines)

# ============================================================================
# 4. FIXED MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function with enhanced visualizations"""
    print(" F1 Driver Performance Analysis System")
    print("=" * 50)
    
    # Initialize pipeline and analyzer
    pipeline = OpenF1DataPipeline(cache_dir="./f1_data_cache")
    analyzer = DriverPerformanceAnalyzer(pipeline)
    visualizer = PerformanceVisualizer()
    
    # Analyze 2024 season
    print("\n1. Analyzing 2024 season...")
    scores_2024 = analyzer.calculate_composite_score(2024)
    
    if scores_2024:
        # ADD DRIVER NAMES HERE
        driver_name_mapping = {
            1: "Max Verstappen", 44: "Lewis Hamilton", 16: "Charles Leclerc",
            4: "Lando Norris", 55: "Carlos Sainz", 11: "Sergio Perez",
            63: "George Russell", 81: "Oscar Piastri", 14: "Fernando Alonso",
            18: "Lance Stroll", 22: "Yuki Tsunoda", 3: "Daniel Ricciardo",
            27: "Nico Hulkenberg", 20: "Kevin Magnussen", 23: "Alexander Albon",
            2: "Logan Sargeant", 24: "Zhou Guanyu", 77: "Valtteri Bottas",
            31: "Esteban Ocon", 10: "Pierre Gasly", 50: "Oliver Bearman"
        }
        
        # Add names and teams to scores
        for driver_num, score_data in scores_2024.items():
            if driver_num in driver_name_mapping:
                score_data['driver_name'] = driver_name_mapping[driver_num]
            else:
                score_data['driver_name'] = f"Driver {driver_num}"
            
            # Add team info
            team_mapping = {
                1: "Red Bull", 11: "Red Bull",
                44: "Mercedes", 63: "Mercedes",
                16: "Ferrari", 55: "Ferrari",
                4: "McLaren", 81: "McLaren",
                14: "Aston Martin", 18: "Aston Martin",
                22: "RB", 3: "RB",
                27: "Haas", 20: "Haas",
                23: "Williams", 2: "Williams",
                24: "Sauber", 77: "Sauber",
                31: "Alpine", 10: "Alpine",
                50: "Haas"
            }
            score_data['team'] = team_mapping.get(driver_num, "Unknown")
        
        # Display simple rankings with names
        print("\n" + "="*50)
        print("2024 DRIVER RANKINGS:")
        print("="*50)
        
        sorted_drivers = sorted(scores_2024.items(), 
                              key=lambda x: x[1]['composite_score'], 
                              reverse=True)
        
        for rank, (driver_num, scores_dict) in enumerate(sorted_drivers[:15], 1):
            name = scores_dict['driver_name']
            team = scores_dict.get('team', 'Unknown')
            score = scores_dict['composite_score']
            print(f"{rank:2}. {name:20} ({team:15}) - Score: {score:.1f}")
        
        # Display full report
        report = visualizer.create_performance_report(scores_2024, 2024)
        print("\n" + report)
        
        # Create visualizations WITH NAMES
        print("\n Generating visualizations...")
        visualizer.plot_composite_scores(scores_2024, 2024)
        
        # Show radar chart for top 3 drivers
        sorted_drivers = sorted(scores_2024.items(), 
                              key=lambda x: x[1]['composite_score'], 
                              reverse=True)
        
        for i, (driver_num, driver_scores) in enumerate(sorted_drivers[:3], 1):
            driver_name = driver_scores['driver_name']
            print(f"\n Generating radar chart for #{i}: {driver_name}")
            visualizer.plot_radar_chart(driver_scores, f"Rank #{i}: {driver_name}")
        
        # Save results to CSV
        results_df = pd.DataFrame.from_dict(scores_2024, orient='index')
        results_df.to_csv(f"driver_performance_2024.csv")
        print(f"\n Results saved to 'driver_performance_2024.csv'")
        
        # Save rankings separately
        ranking_data = []
        for rank, (driver_num, scores_dict) in enumerate(sorted_drivers, 1):
            ranking_data.append({
                'rank': rank,
                'driver_number': driver_num,
                'driver_name': scores_dict['driver_name'],
                'team': scores_dict.get('team', 'Unknown'),
                'composite_score': scores_dict['composite_score'],
                'quali_score': scores_dict.get('quali_score', 0),
                'pace_score': scores_dict.get('pace_score', 0),
                'consistency_score': scores_dict.get('consistency_score', 0),
                'racecraft_score': scores_dict.get('racecraft_score', 0),
                'reliability_score': scores_dict.get('reliability_score', 0)
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_csv(f"driver_rankings_2024.csv", index=False)
        print(f"    Rankings saved to 'driver_rankings_2024.csv'")
        
    else:
        print("Could not calculate scores for 2024")
    
    # For prediction phase
    print("\n" + "="*50)
    print("Next Phase: 2025 Champion Prediction")
    print("="*50)
    print("\nTo implement prediction for 2025, we need to:")
    print("1. Collect historical data (2020-2024)")
    print("2. Engineer predictive features")
    print("3. Train ML models (XGBoost, Random Forest)")
    print("4. Validate using backtesting")
    
    return scores_2024

# ============================================================================
# 5. RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    scores = main()