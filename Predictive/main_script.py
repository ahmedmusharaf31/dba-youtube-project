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
import sqlite3
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. SQLITE CACHE CLASS
class F1DataCache:
    """SQLite database for caching F1 data"""
    
    def __init__(self, db_path="f1_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            endpoint TEXT,
            params TEXT,
            data TEXT,
            timestamp DATETIME,
            PRIMARY KEY (endpoint, params)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS driver_scores (
            year INTEGER,
            driver_number INTEGER,
            driver_name TEXT,
            team TEXT,
            composite_score REAL,
            quali_score REAL,
            pace_score REAL,
            consistency_score REAL,
            racecraft_score REAL,
            reliability_score REAL,
            timestamp DATETIME,
            PRIMARY KEY (year, driver_number)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS season_data (
            year INTEGER,
            meeting_key TEXT,
            meeting_name TEXT,
            country_name TEXT,
            session_data TEXT,
            timestamp DATETIME,
            PRIMARY KEY (year, meeting_key)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def cache_api_response(self, endpoint, params, data):
        """Cache API response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        params_str = json.dumps(params) if params else "{}"
        data_str = json.dumps(data) if data else "[]"
        
        cursor.execute('''
        INSERT OR REPLACE INTO api_cache 
        (endpoint, params, data, timestamp) 
        VALUES (?, ?, ?, ?)
        ''', (endpoint, params_str, data_str, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_cached_api_response(self, endpoint, params, max_age_hours=24):
        """Get cached API response if not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        params_str = json.dumps(params) if params else "{}"
        
        cursor.execute('''
        SELECT data, timestamp FROM api_cache 
        WHERE endpoint = ? AND params = ?
        ''', (endpoint, params_str))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data_str, timestamp_str = result
            timestamp = datetime.fromisoformat(timestamp_str)
            
            # Check if cache is expired
            if datetime.now() - timestamp < timedelta(hours=max_age_hours):
                return json.loads(data_str)
        
        return None
    
    def cache_driver_scores(self, year, driver_scores):
        """Cache calculated driver scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for driver_num, scores in driver_scores.items():
            cursor.execute('''
            INSERT OR REPLACE INTO driver_scores 
            (year, driver_number, driver_name, team, composite_score, 
             quali_score, pace_score, consistency_score, racecraft_score, 
             reliability_score, timestamp) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                year,
                driver_num,
                scores.get('driver_name', f'Driver {driver_num}'),
                scores.get('team', 'Unknown'),
                scores.get('composite_score', 0),
                scores.get('quali_score', 0),
                scores.get('pace_score', 0),
                scores.get('consistency_score', 0),
                scores.get('racecraft_score', 0),
                scores.get('reliability_score', 0),
                datetime.now()
            ))
        
        conn.commit()
        conn.close()
    
    def get_cached_driver_scores(self, year):
        """Get cached driver scores for a year"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM driver_scores WHERE year = ?
        ORDER BY composite_score DESC
        ''', (year,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        # Convert to dictionary format
        driver_scores = {}
        for row in rows:
            driver_num = row[1]
            driver_scores[driver_num] = {
                'driver_number': driver_num,
                'driver_name': row[2],
                'team': row[3],
                'composite_score': row[4],
                'quali_score': row[5],
                'pace_score': row[6],
                'consistency_score': row[7],
                'racecraft_score': row[8],
                'reliability_score': row[9]
            }
        
        return driver_scores
    
    def clear_old_cache(self, days_old=30):
        """Clear cache older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        cursor.execute('DELETE FROM api_cache WHERE timestamp < ?', (cutoff_date,))
        cursor.execute('DELETE FROM driver_scores WHERE timestamp < ?', (cutoff_date,))
        cursor.execute('DELETE FROM season_data WHERE timestamp < ?', (cutoff_date,))
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        print(f"Cleared {deleted} old cache entries")
        return deleted

        
        
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



class CachedOpenF1DataPipeline(OpenF1DataPipeline):
    """OpenF1 pipeline with SQLite caching"""
    
    def __init__(self, cache_dir="./f1_cache", db_path="f1_data.db"):
        super().__init__(cache_dir)
        self.db_cache = F1DataCache(db_path)
    
    def _make_request(self, endpoint, params=None, use_cache=True, cache_hours=24):
        """Make API request with database caching"""
        # First check database cache
        if use_cache:
            cached_data = self.db_cache.get_cached_api_response(endpoint, params, cache_hours)
            if cached_data is not None:
                print(f"[CACHE] Using DB cache: {endpoint}")
                return pd.DataFrame(cached_data)
        
        # Then check file cache
        cache_key = f"{endpoint}_{hash(frozenset(params.items()) if params else '')}.json"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if use_cache and os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                data = json.load(f)
            print(f"[CACHE] Using file cache: {endpoint}")
            
            # Also store in database
            self.db_cache.cache_api_response(endpoint, params, data)
            
            return pd.DataFrame(data) if data else pd.DataFrame()
        
        # Make API call
        try:
            print(f" Fetching from API: {endpoint}")
            response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            # Cache in file
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            # Cache in database
            self.db_cache.cache_api_response(endpoint, params, data)
            
            return pd.DataFrame(data) if data else pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {endpoint}: {e}")
            return pd.DataFrame()

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


class CachedDriverPerformanceAnalyzer(DriverPerformanceAnalyzer):
    """Analyzer with score caching"""
    
    def __init__(self, data_pipeline):
        super().__init__(data_pipeline)
        if hasattr(data_pipeline, 'db_cache'):
            self.db_cache = data_pipeline.db_cache
        else:
            self.db_cache = F1DataCache()
    
    def calculate_composite_score(self, year, use_cache=True):
        """Calculate or retrieve cached composite scores"""
        # Check cache first
        if use_cache:
            cached_scores = self.db_cache.get_cached_driver_scores(year)
            if cached_scores is not None:
                print(f" Using cached scores for {year}")
                return cached_scores
        
        # Calculate fresh scores
        print(f"Calculating scores for {year}...")
        scores = super().calculate_composite_score(year)
        
        # Cache the results
        if scores:
            self.db_cache.cache_driver_scores(year, scores)
        
        return scores
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
# 5. PREDICTIVE ANALYTICS MODULE
# ============================================================================

class ChampionPredictor:
    """Predicts next season's champion using historical data and ML models"""
    
    def __init__(self, data_pipeline):
        self.pipeline = data_pipeline
        # Use cached analyzer if pipeline supports caching
        if hasattr(data_pipeline, 'db_cache'):
            self.analyzer = CachedDriverPerformanceAnalyzer(data_pipeline)
        else:
            self.analyzer = DriverPerformanceAnalyzer(data_pipeline)
        self.models = {}
        self.feature_importance = {}
        self.imputer = None
        self.scaler = None
        
    def collect_historical_data(self, start_year=2020, end_year=2024):
        """Collect historical data for multiple seasons"""
        print(f"\n Collecting historical data ({start_year}-{end_year})...")
        
        historical_data = {}
        
        for year in range(start_year, end_year + 1):
            print(f"  Processing {year} season...")
            scores = self.analyzer.calculate_composite_score(year)
            
            if scores:
                # Add driver names and teams
                driver_name_mapping = {
                    1: "Max Verstappen", 44: "Lewis Hamilton", 16: "Charles Leclerc",
                    4: "Lando Norris", 55: "Carlos Sainz", 11: "Sergio Perez",
                    63: "George Russell", 81: "Oscar Piastri", 14: "Fernando Alonso",
                    18: "Lance Stroll", 22: "Yuki Tsunoda", 3: "Daniel Ricciardo",
                    27: "Nico Hulkenberg", 20: "Kevin Magnussen", 23: "Alexander Albon",
                    2: "Logan Sargeant", 24: "Zhou Guanyu", 77: "Valtteri Bottas",
                    31: "Esteban Ocon", 10: "Pierre Gasly", 50: "Oliver Bearman"
                }
                
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
                
                for driver_num, score_data in scores.items():
                    if driver_num in driver_name_mapping:
                        score_data['driver_name'] = driver_name_mapping[driver_num]
                        score_data['team'] = team_mapping.get(driver_num, "Unknown")
                    else:
                        score_data['driver_name'] = f"Driver {driver_num}"
                        score_data['team'] = "Unknown"
                
                historical_data[year] = scores
            else:
                print(f"   No data for {year}")
        
        return historical_data
    
    def create_training_dataset(self, historical_data):
        """Create training dataset from historical data"""
        print("\n Creating training dataset...")
        
        features = []
        labels = []
        driver_info = []
        
        # For each season (except the last one), predict next season's champion
        years = sorted(historical_data.keys())
        
        for i in range(len(years) - 1):
            current_year = years[i]
            next_year = years[i + 1]
            
            print(f"  Using {current_year} to predict {next_year} champion...")
            
            current_data = historical_data[current_year]
            next_data = historical_data[next_year]
            
            # Find champion of next season
            if not next_data:
                continue
            
            next_champion = max(next_data.items(), 
                              key=lambda x: x[1].get('composite_score', 0))[0]
            
            # For each driver in current season, create features
            for driver_num, scores in current_data.items():
                driver_name = scores.get('driver_name', f"Driver {driver_num}")
                team = scores.get('team', 'Unknown')
                
                # Create feature vector
                feature_vector = [
                    scores.get('composite_score', 0),
                    scores.get('quali_score', 0),
                    scores.get('pace_score', 0),
                    scores.get('consistency_score', 0),
                    scores.get('racecraft_score', 0),
                    scores.get('reliability_score', 0),
                    # Additional derived features
                    scores.get('component_scores', {}).get('avg_quali_position', 20),
                    scores.get('component_scores', {}).get('avg_pace_gap', 1.0),
                    scores.get('component_scores', {}).get('avg_consistency', 0.5),
                    scores.get('component_scores', {}).get('avg_positions_gained', 0),
                    scores.get('component_scores', {}).get('finish_rate', 0.5)
                ]
                
                # Label: 1 if champion next season, 0 otherwise
                label = 1 if driver_num == next_champion else 0
                
                features.append(feature_vector)
                labels.append(label)
                driver_info.append({
                    'driver_num': driver_num,
                    'driver_name': driver_name,
                    'team': team,
                    'year': current_year,
                    'next_year': next_year
                })
        
        return np.array(features), np.array(labels), driver_info
    
    def train_models(self, X, y):
        """Train multiple ML models"""
        print("\n Training machine learning models...")

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        # Handle missing values and preprocess data
        print("  Preprocessing data...")
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Split data (remove stratify to handle imbalanced classes)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=3, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = model.feature_importances_
            
            print(f"    Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        self.models = results
        return results
    
    def predict_2025_champion(self, current_season_data):
        """Predict 2025 champion based on 2024 data"""
        print("\n Predicting 2025 Formula 1 World Champion...")
        print("=" * 60)
        
        if not self.models:
            print(" No trained models available. Training models first...")
            return None
        
        # Prepare 2024 data for prediction
        features_2024 = []
        driver_info_2024 = []
        
        for driver_num, scores in current_season_data.items():
            driver_name = scores.get('driver_name', f"Driver {driver_num}")
            team = scores.get('team', 'Unknown')
            
            feature_vector = [
                scores.get('composite_score', 0),
                scores.get('quali_score', 0),
                scores.get('pace_score', 0),
                scores.get('consistency_score', 0),
                scores.get('racecraft_score', 0),
                scores.get('reliability_score', 0),
                scores.get('component_scores', {}).get('avg_quali_position', 20),
                scores.get('component_scores', {}).get('avg_pace_gap', 1.0),
                scores.get('component_scores', {}).get('avg_consistency', 0.5),
                scores.get('component_scores', {}).get('avg_positions_gained', 0),
                scores.get('component_scores', {}).get('finish_rate', 0.5)
            ]
            
            features_2024.append(feature_vector)
            driver_info_2024.append({
                'driver_num': driver_num,
                'driver_name': driver_name,
                'team': team,
                'composite_score': scores.get('composite_score', 0)
            })
        
        X_2024 = np.array(features_2024)

        # Apply the same preprocessing as training data
        X_2024_imputed = self.imputer.transform(X_2024)
        X_2024_scaled = self.scaler.transform(X_2024_imputed)

        # Get predictions from all models
        predictions = {}

        for model_name, model_data in self.models.items():
            model = model_data['model']

            # Get probabilities
            probabilities = model.predict_proba(X_2024_scaled)[:, 1]
            
            # Find driver with highest probability
            champion_idx = np.argmax(probabilities)
            champion_prob = probabilities[champion_idx]
            champion_info = driver_info_2024[champion_idx]
            
            predictions[model_name] = {
                'champion_name': champion_info['driver_name'],
                'champion_team': champion_info['team'],
                'probability': champion_prob,
                'composite_score': champion_info['composite_score'],
                'all_probabilities': probabilities,
                'driver_info': driver_info_2024
            }
        
        return predictions
    
    def display_predictions(self, predictions):
        """Display prediction results"""
        if not predictions:
            print("No predictions available")
            return
        
        print("\n 2025 CHAMPION PREDICTIONS")
        print("=" * 60)
        
        # Display each model's prediction
        for model_name, pred in predictions.items():
            print(f"\n{model_name}:")
            print(f"  Champion: {pred['champion_name']} ({pred['champion_team']})")
            print(f"  Probability: {pred['probability']:.1%}")
            print(f"  2024 Composite Score: {pred['composite_score']:.1f}")
        
        # Consensus prediction
        print("\n" + "=" * 60)
        print(" CONSENSUS PREDICTION")
        print("=" * 60)
        
        # Calculate weighted average
        champion_votes = {}
        for model_name, pred in predictions.items():
            champion = pred['champion_name']
            prob = pred['probability']
            
            if champion not in champion_votes:
                champion_votes[champion] = {'votes': 0, 'total_prob': 0}
            
            champion_votes[champion]['votes'] += 1
            champion_votes[champion]['total_prob'] += prob
        
        # Find consensus
        if champion_votes:
            consensus = max(champion_votes.items(), 
                          key=lambda x: (x[1]['votes'], x[1]['total_prob']))
            
            champion_name = consensus[0]
            votes = consensus[1]['votes']
            avg_prob = consensus[1]['total_prob'] / votes
            
            print(f"\nBased on {len(predictions)} models:")
            print(f"   Most likely 2025 Champion: {champion_name}")
            print(f"   Model votes: {votes}/{len(predictions)}")
            print(f"   Average probability: {avg_prob:.1%}")
        
        # Top 3 contenders
        print("\n" + "=" * 60)
        print(" TOP 3 CONTENDERS FOR 2025")
        print("=" * 60)
        
        # Aggregate probabilities across all models
        driver_probs = {}
        for model_name, pred in predictions.items():
            for i, driver_info in enumerate(pred['driver_info']):
                driver_name = driver_info['driver_name']
                prob = pred['all_probabilities'][i]
                
                if driver_name not in driver_probs:
                    driver_probs[driver_name] = []
                
                driver_probs[driver_name].append(prob)
        
        # Calculate average probabilities
        avg_probs = []
        for driver_name, probs in driver_probs.items():
            avg_prob = np.mean(probs)
            avg_probs.append((driver_name, avg_prob))
        
        # Sort and display top 3
        avg_probs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (driver_name, prob) in enumerate(avg_probs[:3], 1):
            print(f"{i}. {driver_name}: {prob:.1%}")
    
    def create_prediction_report(self, predictions, output_file="2025_champion_prediction.txt"):
        """Create detailed prediction report"""
        with open(output_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("F1 2025 WORLD CHAMPION PREDICTION REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("PREDICTION METHODOLOGY\n")
            f.write("-" * 40 + "\n")
            f.write("Models trained on historical data (2020-2024) to predict next season champion.\n")
            f.write("Features used: Composite score and all component scores from current season.\n")
            f.write("Models: Random Forest, Gradient Boosting, Logistic Regression\n\n")

            f.write("MODEL PREDICTIONS\n")
            f.write("-" * 40 + "\n")

            for model_name, pred in predictions.items():
                f.write(f"{model_name}:\n")
                f.write(f"  Champion: {pred['champion_name']} ({pred['champion_team']})\n")
                f.write(f"  Probability: {pred['probability']:.1%}\n")
                f.write(f"  2024 Composite Score: {pred['composite_score']:.1f}\n\n")

            # Consensus
            champion_votes = {}
            for model_name, pred in predictions.items():
                champion = pred['champion_name']
                if champion not in champion_votes:
                    champion_votes[champion] = 0
                champion_votes[champion] += 1

            if champion_votes:
                consensus = max(champion_votes.items(), key=lambda x: x[1])
                f.write("CONSENSUS PREDICTION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Most likely 2025 Champion: {consensus[0]}\n")
                f.write(f"Model votes: {consensus[1]}/{len(predictions)}\n\n")

            f.write("KEY FACTORS FOR CHAMPIONSHIP SUCCESS\n")
            f.write("-" * 40 + "\n")
            f.write("1. Race Pace (30%): Most important factor in predictions\n")
            f.write("2. Consistency (20%): Regular top finishes are crucial\n")
            f.write("3. Qualifying Performance (25%): Grid position matters\n")
            f.write("4. Reliability (10%): Must finish races consistently\n")
            f.write("5. Racecraft (15%): Overtaking and race management\n")

        print(f"✓ Prediction report saved to '{output_file}'")

    def predict_driver_championship(self):
        """Predict driver championship winners"""
        if not hasattr(self, 'models') or not self.models:
            return None

        # Get current season data (assuming 2024)
        current_scores = self.analyzer.calculate_composite_score(2024, use_cache=True)
        if not current_scores:
            return None

        # Add driver info
        driver_name_mapping = {
            1: "Max Verstappen", 44: "Lewis Hamilton", 16: "Charles Leclerc",
            4: "Lando Norris", 55: "Carlos Sainz", 11: "Sergio Perez",
            63: "George Russell", 81: "Oscar Piastri", 14: "Fernando Alonso",
            18: "Lance Stroll", 22: "Yuki Tsunoda", 3: "Daniel Ricciardo",
            27: "Nico Hulkenberg", 20: "Kevin Magnussen", 23: "Alexander Albon",
            2: "Logan Sargeant", 24: "Zhou Guanyu", 77: "Valtteri Bottas",
            31: "Esteban Ocon", 10: "Pierre Gasly", 50: "Oliver Bearman"
        }

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

        for driver_num, score_data in current_scores.items():
            score_data['driver_name'] = driver_name_mapping.get(driver_num, f"Driver {driver_num}")
            score_data['team'] = team_mapping.get(driver_num, "Unknown")

        # Prepare features
        features = []
        driver_info = []

        for driver_num, scores in current_scores.items():
            feature_vector = [
                scores.get('composite_score', 0),
                scores.get('quali_score', 0),
                scores.get('pace_score', 0),
                scores.get('consistency_score', 0),
                scores.get('racecraft_score', 0),
                scores.get('reliability_score', 0),
                scores.get('component_scores', {}).get('avg_quali_position', 20),
                scores.get('component_scores', {}).get('avg_pace_gap', 1.0),
                scores.get('component_scores', {}).get('avg_consistency', 0.5),
                scores.get('component_scores', {}).get('avg_positions_gained', 0),
                scores.get('component_scores', {}).get('finish_rate', 0.5)
            ]

            features.append(feature_vector)
            driver_info.append({
                'driver_num': driver_num,
                'driver_name': scores.get('driver_name'),
                'team': scores.get('team')
            })

        X = np.array(features)

        # Get predictions from best model (Random Forest)
        if 'Random Forest' in self.models:
            model = self.models['Random Forest']['model']
            probabilities = model.predict_proba(X)[:, 1]

            # Sort by probability
            predictions = []
            for i, prob in enumerate(probabilities):
                predictions.append({
                    'driver_num': driver_info[i]['driver_num'],
                    'probability': prob
                })

            predictions.sort(key=lambda x: x['probability'], reverse=True)
            return {pred['driver_num']: pred['probability'] for pred in predictions}

        return None

    def predict_constructor_championship(self):
        """Predict constructor championship winners"""
        if not hasattr(self, 'models') or not self.models:
            return None

        # Get current season data
        current_scores = self.analyzer.calculate_composite_score(2024, use_cache=True)
        if not current_scores:
            return None

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

        # Calculate team averages
        team_scores = {}
        for driver_num, scores in current_scores.items():
            team = team_mapping.get(driver_num, "Unknown")
            if team not in team_scores:
                team_scores[team] = []
            team_scores[team].append(scores.get('composite_score', 0))

        # Calculate team average scores
        team_averages = {}
        for team, scores in team_scores.items():
            team_averages[team] = np.mean(scores)

        # Sort teams by average score
        sorted_teams = sorted(team_averages.items(), key=lambda x: x[1], reverse=True)

        # Convert to probability-like scores (normalized)
        if sorted_teams:
            max_score = sorted_teams[0][1]
            min_score = sorted_teams[-1][1]
            if max_score != min_score:
                team_probs = {}
                for team, score in sorted_teams:
                    # Normalize to 0-1 range
                    prob = (score - min_score) / (max_score - min_score)
                    team_probs[team] = prob
                return team_probs

        return None

    def predict_race_outcomes(self):
        """Predict outcomes for upcoming races"""
        # This is a simplified prediction - in reality would need more sophisticated modeling
        # For now, return mock predictions for upcoming races

        upcoming_races = [
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix"
        ]

        predictions = {}

        # Get current scores for predictions
        current_scores = self.analyzer.calculate_composite_score(2024, use_cache=True)
        if not current_scores:
            return None

        # Sort drivers by composite score
        sorted_drivers = sorted(current_scores.items(),
                              key=lambda x: x[1].get('composite_score', 0),
                              reverse=True)

        for race in upcoming_races:
            # Simple prediction: top 3 based on current form
            top_3 = []
            win_probabilities = {}

            for i, (driver_num, scores) in enumerate(sorted_drivers[:10]):  # Top 10 drivers
                if i < 3:
                    top_3.append(driver_num)

                # Assign win probabilities based on ranking
                if i == 0:
                    win_probabilities[driver_num] = 0.35
                elif i == 1:
                    win_probabilities[driver_num] = 0.25
                elif i == 2:
                    win_probabilities[driver_num] = 0.15
                elif i < 6:
                    win_probabilities[driver_num] = 0.05
                else:
                    win_probabilities[driver_num] = 0.01

            predictions[race] = {
                'top_3': top_3,
                'win_probabilities': win_probabilities
            }

        return predictions

    def get_model_metrics(self):
        """Get metrics from trained models"""
        if not hasattr(self, 'models') or not self.models:
            return None

        # Return metrics from Random Forest model (as example)
        if 'Random Forest' in self.models:
            model_data = self.models['Random Forest']
            return {
                'accuracy': model_data.get('accuracy', 0),
                'precision': model_data.get('precision', 0),
                'recall': model_data.get('recall', 0),
                'f1_score': model_data.get('f1_score', 0) if 'f1_score' in model_data else 0
            }

        return None

    def save_prediction_probabilities(self, predictions, current_season_data, output_file="2025_predictions.csv"):
        """Save prediction probabilities to CSV file"""
        import pandas as pd

        if not predictions:
            print("No predictions to save")
            return

        # Collect all driver probabilities across models
        driver_probs = {}
        driver_info = {}

        for model_name, pred in predictions.items():
            for i, driver_info_item in enumerate(pred['driver_info']):
                driver_num = driver_info_item['driver_num']
                driver_name = driver_info_item['driver_name']
                team = driver_info_item['team']
                prob = pred['all_probabilities'][i]

                if driver_num not in driver_probs:
                    driver_probs[driver_num] = {}
                    driver_info[driver_num] = {
                        'driver_name': driver_name,
                        'team': team,
                        'composite_score': current_season_data.get(driver_num, {}).get('composite_score', 0)
                    }

                driver_probs[driver_num][model_name] = prob

        # Create DataFrame
        data = []
        for driver_num, probs in driver_probs.items():
            row = {
                'driver_number': driver_num,
                'driver_name': driver_info[driver_num]['driver_name'],
                'team': driver_info[driver_num]['team'],
                'composite_score': driver_info[driver_num]['composite_score']
            }

            # Add probabilities from each model
            for model_name in predictions.keys():
                row[f'{model_name.lower().replace(" ", "_")}_probability'] = probs.get(model_name, 0)

            # Calculate average probability
            avg_prob = sum(probs.values()) / len(probs)
            row['average_probability'] = avg_prob

            data.append(row)

        # Sort by average probability
        data.sort(key=lambda x: x['average_probability'], reverse=True)

        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f" Prediction probabilities saved to '{output_file}'")

# ============================================================================
# 4. FIXED MAIN FUNCTION
# ============================================================================

def main():
    """Main execution function with caching and prediction"""
    print("F1 Driver Performance Analysis & Prediction System")
    print("=" * 60)
    
    # Initialize CACHED pipeline and analyzer
    pipeline = CachedOpenF1DataPipeline(cache_dir="./f1_data_cache")
    analyzer = CachedDriverPerformanceAnalyzer(pipeline)
    visualizer = PerformanceVisualizer()
    
    # Optional: Clear old cache (hardcoded for testing)
    clear_cache = True  # Hardcoded to 'y' for testing
    if clear_cache:
        pipeline.db_cache.clear_old_cache(30)
    
    # ============================================================
    # PART 1: ANALYZE 2024 SEASON (WITH CACHING)
    # ============================================================
    print("\n" + "="*50)
    print("1. ANALYZING 2024 SEASON")
    print("="*50)
    
    scores_2024 = analyzer.calculate_composite_score(2024, use_cache=True)
    
    if not scores_2024:
        print("Could not calculate scores for 2024")
        return
    
    # Add driver names and teams
    driver_name_mapping = {
        1: "Max Verstappen", 44: "Lewis Hamilton", 16: "Charles Leclerc",
        4: "Lando Norris", 55: "Carlos Sainz", 11: "Sergio Perez",
        63: "George Russell", 81: "Oscar Piastri", 14: "Fernando Alonso",
        18: "Lance Stroll", 22: "Yuki Tsunoda", 3: "Daniel Ricciardo",
        27: "Nico Hulkenberg", 20: "Kevin Magnussen", 23: "Alexander Albon",
        2: "Logan Sargeant", 24: "Zhou Guanyu", 77: "Valtteri Bottas",
        31: "Esteban Ocon", 10: "Pierre Gasly", 50: "Oliver Bearman"
    }
    
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
    
    for driver_num, score_data in scores_2024.items():
        if driver_num in driver_name_mapping:
            score_data['driver_name'] = driver_name_mapping[driver_num]
            score_data['team'] = team_mapping.get(driver_num, "Unknown")
        else:
            score_data['driver_name'] = f"Driver {driver_num}"
            score_data['team'] = "Unknown"
    
    # Display rankings
    print("\n" + "="*60)
    print(" 2024 DRIVER RANKINGS")
    print("="*60)
    
    sorted_drivers = sorted(scores_2024.items(), 
                          key=lambda x: x[1]['composite_score'], 
                          reverse=True)
    
    print(f"\n{'Rank':<6} {'Driver':<20} {'Team':<15} {'Score':<10} {'Q':<6} {'P':<6} {'C':<6} {'R':<6} {'Rel':<6}")
    print("-" * 75)
    
    for rank, (driver_num, scores_dict) in enumerate(sorted_drivers[:15], 1):
        name = scores_dict['driver_name']
        team = scores_dict.get('team', 'Unknown')
        score = scores_dict['composite_score']
        
        print(f"{rank:<6} {name:<20} {team:<15} {score:<10.1f} "
              f"{scores_dict.get('quali_score', 0):<6.0f} "
              f"{scores_dict.get('pace_score', 0):<6.0f} "
              f"{scores_dict.get('consistency_score', 0):<6.0f} "
              f"{scores_dict.get('racecraft_score', 0):<6.0f} "
              f"{scores_dict.get('reliability_score', 0):<6.0f}")
    
    # Save results
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
    print(f" Rankings saved to 'driver_rankings_2024.csv'")
    
    # ============================================================
    # PART 2: PREDICTIVE ANALYTICS
    # ============================================================
    print("\n" + "="*60)
    print("2. PREDICTIVE ANALYTICS: 2025 CHAMPION FORECAST")
    print("="*60)
    
    run_prediction = True  # Hardcoded to 'y' for testing
    
    if run_prediction:
        try:
            # Check if scikit-learn is installed
            import sklearn
            
            # Initialize predictor
            predictor = ChampionPredictor(pipeline)
            
            # Collect historical data (with caching)
            print("\nCollecting historical data (2020-2024)...")
            historical_data = {}
            
            for year in range(2020, 2025):
                if year == 2024:
                    historical_data[year] = scores_2024
                else:
                    scores = analyzer.calculate_composite_score(year, use_cache=True)
                    if scores:
                        # Add names to historical data
                        for driver_num, score_data in scores.items():
                            if driver_num in driver_name_mapping:
                                score_data['driver_name'] = driver_name_mapping[driver_num]
                                score_data['team'] = team_mapping.get(driver_num, "Unknown")
                        historical_data[year] = scores
            
            if len(historical_data) >= 2:
                # Create training dataset
                X, y, driver_info = predictor.create_training_dataset(historical_data)
                
                print(f"\nDataset created:")
                print(f"  Samples: {X.shape[0]}")
                print(f"  Features: {X.shape[1]}")
                print(f"  Champions in dataset: {sum(y)}")
                
                # Train models
                results = predictor.train_models(X, y)
                
                # Predict 2025 champion
                predictions = predictor.predict_2025_champion(scores_2024)
                
                if predictions:
                    # Display predictions
                    predictor.display_predictions(predictions)
                    
                    # Create detailed report
                    predictor.create_prediction_report(predictions)
                    
                    # Save prediction probabilities
                    predictor.save_prediction_probabilities(predictions, scores_2024)
                else:
                    print(" Could not generate predictions")
            else:
                print(" Not enough historical data for prediction")
                print("   Need at least 2 seasons of data")
                
        except ImportError:
            print("\n scikit-learn not installed. Skipping prediction.")
            print("   Install with: pip install scikit-learn")
        except Exception as e:
            print(f"\n Prediction error: {e}")
            print("   Skipping prediction module")
    
    # ============================================================
    # PART 3: VISUALIZATIONS
    # ============================================================
    print("\n" + "="*60)
    print("3. VISUALIZATIONS")
    print("="*60)
    
    show_viz = input("\nGenerate visualizations? (y/n): ").lower() == 'y'
    
    if show_viz:
        try:
            print("\n📊 Generating visualizations...")
            visualizer.plot_composite_scores(scores_2024, 2024)
            
            # Radar chart for top driver
            top_driver = sorted_drivers[0][1]
            top_name = top_driver['driver_name']
            print(f"\n🎯 Generating radar chart for top driver: {top_name}")
            visualizer.plot_radar_chart(top_driver, f"2024 Top Performer: {top_name}")
            
        except Exception as e:
            print(f" Visualization error: {e}")
    
    # ============================================================
    # PART 4: CACHE STATISTICS
    # ============================================================
    print("\n" + "="*60)
    print("4. CACHE STATISTICS")
    print("="*60)
    
    # Check cache file size
    if os.path.exists("f1_data.db"):
        size_mb = os.path.getsize("f1_data.db") / (1024 * 1024)
        print(f"\n📊 Database cache: {size_mb:.2f} MB")
    
    if os.path.exists("./f1_data_cache"):
        cache_files = len([f for f in os.listdir("./f1_data_cache") if f.endswith('.json')])
        print(f"📊 JSON cache files: {cache_files}")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)
    
    print("\n📁 Generated files:")
    print(f"  • driver_performance_2024.csv")
    print(f"  • driver_rankings_2024.csv")
    
    if run_prediction:
        print(f"  • 2025_champion_prediction.txt")
        print(f"  • 2025_predictions.csv")
    
    print(f"  • f1_driver_scores_2024.png")
    print(f"  • radar_chart_*.png")
    
    print("\n💾 Next time you run this script:")
    print("  • 2024 data will load instantly from cache")
    print("  • Historical data will load from cache")
    
    return scores_2024

if __name__ == "__main__":
    scores = main()