"""
Configuration file for F1 YouTube Descriptive Analytics
Contains API credentials, driver/team mappings, and analysis parameters
"""
from pathlib import Path
import os

# =============================================================================
# Project Paths
# =============================================================================
SRC_DIR = Path(__file__).parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

# CSV file paths
VIDEOS_CSV = RAW_DATA_DIR / "f1_youtube_videos.csv"
COMMENTS_CSV = RAW_DATA_DIR / "f1_youtube_comments.csv"
CLEAN_DATASET_CSV = PROCESSED_DATA_DIR / "f1_clean_dataset.csv"
FEATURES_CSV = PROCESSED_DATA_DIR / "f1_final_features.csv"

# =============================================================================
# YouTube API Credentials
# =============================================================================
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyC8zaa9CfAXLNkblqmd7kp8nnOmb6KSvl0")

# OAuth 2.0 credentials (for advanced operations if needed)
CLIENT_ID = "751197645124-j5br24fuh7nfr4tdhojc265vigku92km.apps.googleusercontent.com"
CLIENT_SECRET = "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"

# Formula 1 Official YouTube Channel ID
F1_CHANNEL_ID = "UCB_qr75-ydFVKSF9Dmo6izg"

# =============================================================================
# Data Extraction Settings
# =============================================================================
MAX_VIDEOS = 200
MAX_COMMENTS_PER_VIDEO = 100
SEASON_START_DATE = "2024-01-01T00:00:00Z"
SEASON_END_DATE = "2024-12-31T23:59:59Z"

# =============================================================================
# Data Paths (legacy string paths for backward compatibility)
# =============================================================================
RAW_DATA_PATH = str(RAW_DATA_DIR) + "/"
PROCESSED_DATA_PATH = str(PROCESSED_DATA_DIR) + "/"

# =============================================================================
# F1 Driver Mappings (2024 Season)
# =============================================================================
DRIVERS = {
    # Red Bull Racing
    "verstappen": {"full_name": "Max Verstappen", "team": "Red Bull", "number": 1, "aliases": ["max", "verstappen", "ver", "mad max", "super max"]},
    "perez": {"full_name": "Sergio Perez", "team": "Red Bull", "number": 11, "aliases": ["sergio", "perez", "per", "checo"]},
    
    # Ferrari
    "leclerc": {"full_name": "Charles Leclerc", "team": "Ferrari", "number": 16, "aliases": ["charles", "leclerc", "lec", "sharl"]},
    "sainz": {"full_name": "Carlos Sainz", "team": "Ferrari", "number": 55, "aliases": ["carlos", "sainz", "sai", "smooth operator"]},
    
    # McLaren
    "norris": {"full_name": "Lando Norris", "team": "McLaren", "number": 4, "aliases": ["lando", "norris", "nor", "landito"]},
    "piastri": {"full_name": "Oscar Piastri", "team": "McLaren", "number": 81, "aliases": ["oscar", "piastri", "pia"]},
    
    # Mercedes
    "hamilton": {"full_name": "Lewis Hamilton", "team": "Mercedes", "number": 44, "aliases": ["lewis", "hamilton", "ham", "goat", "sir lewis"]},
    "russell": {"full_name": "George Russell", "team": "Mercedes", "number": 63, "aliases": ["george", "russell", "rus", "mr saturday"]},
    
    # Aston Martin
    "alonso": {"full_name": "Fernando Alonso", "team": "Aston Martin", "number": 14, "aliases": ["fernando", "alonso", "alo", "el plan", "magic alonso"]},
    "stroll": {"full_name": "Lance Stroll", "team": "Aston Martin", "number": 18, "aliases": ["lance", "stroll", "str"]},
    
    # Alpine
    "gasly": {"full_name": "Pierre Gasly", "team": "Alpine", "number": 10, "aliases": ["pierre", "gasly", "gas"]},
    "ocon": {"full_name": "Esteban Ocon", "team": "Alpine", "number": 31, "aliases": ["esteban", "ocon", "oco"]},
    
    # Williams
    "albon": {"full_name": "Alexander Albon", "team": "Williams", "number": 23, "aliases": ["alex", "albon", "alb"]},
    "sargeant": {"full_name": "Logan Sargeant", "team": "Williams", "number": 2, "aliases": ["logan", "sargeant", "sar"]},
    "colapinto": {"full_name": "Franco Colapinto", "team": "Williams", "number": 43, "aliases": ["franco", "colapinto", "col"]},
    
    # RB (Visa Cash App RB)
    "tsunoda": {"full_name": "Yuki Tsunoda", "team": "RB", "number": 22, "aliases": ["yuki", "tsunoda", "tsu"]},
    "ricciardo": {"full_name": "Daniel Ricciardo", "team": "RB", "number": 3, "aliases": ["daniel", "ricciardo", "ric", "honey badger"]},
    "lawson": {"full_name": "Liam Lawson", "team": "RB", "number": 30, "aliases": ["liam", "lawson", "law"]},
    
    # Kick Sauber
    "bottas": {"full_name": "Valtteri Bottas", "team": "Kick Sauber", "number": 77, "aliases": ["valtteri", "bottas", "bot"]},
    "zhou": {"full_name": "Zhou Guanyu", "team": "Kick Sauber", "number": 24, "aliases": ["zhou", "guanyu", "zho"]},
    
    # Haas
    "magnussen": {"full_name": "Kevin Magnussen", "team": "Haas", "number": 20, "aliases": ["kevin", "magnussen", "mag", "kmag"]},
    "hulkenberg": {"full_name": "Nico Hulkenberg", "team": "Haas", "number": 27, "aliases": ["nico", "hulkenberg", "hul", "hulk"]},
}

# =============================================================================
# F1 Team Mappings
# =============================================================================
TEAMS = {
    "red_bull": {"full_name": "Red Bull Racing", "aliases": ["red bull", "redbull", "rbr", "oracle red bull"]},
    "ferrari": {"full_name": "Scuderia Ferrari", "aliases": ["ferrari", "scuderia", "sf", "prancing horse"]},
    "mclaren": {"full_name": "McLaren", "aliases": ["mclaren", "mcl", "papaya"]},
    "mercedes": {"full_name": "Mercedes-AMG Petronas", "aliases": ["mercedes", "merc", "amg", "silver arrows"]},
    "aston_martin": {"full_name": "Aston Martin", "aliases": ["aston martin", "aston", "am", "amr"]},
    "alpine": {"full_name": "Alpine", "aliases": ["alpine", "alp", "renault"]},
    "williams": {"full_name": "Williams Racing", "aliases": ["williams", "wil"]},
    "rb": {"full_name": "Visa Cash App RB", "aliases": ["rb", "visa cash app", "alphatauri", "alpha tauri", "toro rosso"]},
    "sauber": {"full_name": "Kick Sauber", "aliases": ["sauber", "kick sauber", "alfa romeo", "alfa"]},
    "haas": {"full_name": "MoneyGram Haas", "aliases": ["haas", "moneygram", "haas f1"]},
}

# =============================================================================
# F1 Team Colors (for visualizations)
# =============================================================================
TEAM_COLORS = {
    "red_bull": "#3671C6",
    "ferrari": "#E8002D",
    "mclaren": "#FF8000",
    "mercedes": "#27F4D2",
    "aston_martin": "#229971",
    "alpine": "#FF87BC",
    "williams": "#64C4FF",
    "rb": "#6692FF",
    "sauber": "#52E252",
    "haas": "#B6BABD",
}

# =============================================================================
# F1 Rivalry Pairs (for Rivalry Intensity metric)
# =============================================================================
RIVALRIES = [
    ("verstappen", "norris"),       # 2024 Championship battle
    ("verstappen", "hamilton"),     # Historic rivalry
    ("norris", "piastri"),          # Teammates
    ("leclerc", "sainz"),           # Ferrari teammates
    ("hamilton", "russell"),        # Mercedes teammates
    ("alonso", "hamilton"),         # Historic rivalry
    ("perez", "verstappen"),        # Red Bull teammates
    ("ricciardo", "norris"),        # Former McLaren teammates
    ("gasly", "ocon"),              # French drivers / former teammates
]

# =============================================================================
# NLP Keywords for Topic Analysis
# =============================================================================
F1_KEYWORDS = {
    "racing_action": ["overtake", "overtaking", "pass", "defending", "attack", "battle", "wheel to wheel", "drs"],
    "incidents": ["crash", "collision", "accident", "contact", "spin", "dnf", "retired", "damage"],
    "strategy": ["strategy", "undercut", "overcut", "pit stop", "tyre", "tire", "pit", "box", "medium", "hard", "soft"],
    "penalties": ["penalty", "investigation", "stewards", "track limits", "unsafe release", "warning"],
    "emotions": ["amazing", "incredible", "disaster", "robbed", "unfair", "brilliant", "legendary", "goat"],
    "technical": ["engine", "brake", "suspension", "aero", "downforce", "upgrade", "floor", "wing"],
}

# =============================================================================
# Sentiment Analysis Configuration
# =============================================================================
SENTIMENT_CONFIG = {
    "positive_threshold": 0.05,
    "negative_threshold": -0.05,
}

# =============================================================================
# API Rate Limiting
# =============================================================================
YOUTUBE_QUOTA_LIMIT = 10000  # Daily quota limit
REQUESTS_PER_SECOND = 5     # Rate limiting

# =============================================================================
# Date Range for Analysis (2024 F1 Season) - Legacy
# =============================================================================
SEASON_START = "2024-02-01"  # Pre-season testing
SEASON_END = "2024-12-31"    # Post Abu Dhabi
