"""
Utility functions for F1 YouTube Descriptive Analytics
Contains text processing, sentiment analysis, and helper functions.
"""
import re
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from . import config


# =============================================================================
# NLTK Setup
# =============================================================================

def initialize_nltk():
    """Download required NLTK data if not present."""
    required_packages = ['punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)


# Initialize on import
try:
    initialize_nltk()
except:
    pass  # Will initialize when needed


# =============================================================================
# Sentiment Analysis
# =============================================================================

# Initialize VADER analyzer (singleton)
_vader_analyzer = None

def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Get or create VADER sentiment analyzer."""
    global _vader_analyzer
    if _vader_analyzer is None:
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def analyze_sentiment_vader(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using VADER (optimized for social media).
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with neg, neu, pos, and compound scores
    """
    if not text or not isinstance(text, str):
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    analyzer = get_vader_analyzer()
    return analyzer.polarity_scores(text)


def analyze_sentiment_textblob(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using TextBlob.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with polarity (-1 to 1) and subjectivity (0 to 1)
    """
    if not text or not isinstance(text, str):
        return {'polarity': 0.0, 'subjectivity': 0.0}
    
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }


def classify_sentiment(compound_score: float) -> str:
    """
    Classify sentiment based on VADER compound score.
    
    Args:
        compound_score: VADER compound score (-1 to 1)
        
    Returns:
        'positive', 'negative', or 'neutral'
    """
    if compound_score >= config.SENTIMENT_CONFIG['positive_threshold']:
        return 'positive'
    elif compound_score <= config.SENTIMENT_CONFIG['negative_threshold']:
        return 'negative'
    return 'neutral'


def batch_sentiment_analysis(texts: List[str], show_progress: bool = True) -> Dict[str, List]:
    """
    Perform sentiment analysis on a batch of texts.
    
    Args:
        texts: List of text strings
        show_progress: Whether to show progress
        
    Returns:
        Dictionary with lists of compound scores and labels
    """
    compounds = []
    labels = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 1000 == 0:
            print(f"  Analyzed {i + 1}/{total} texts...")
        
        scores = analyze_sentiment_vader(text)
        compounds.append(scores['compound'])
        labels.append(classify_sentiment(scores['compound']))
    
    return {
        'compound': compounds,
        'label': labels
    }


# =============================================================================
# Driver & Team Detection
# =============================================================================

def detect_drivers(text: str) -> List[str]:
    """
    Detect F1 drivers mentioned in text.
    
    Args:
        text: Text to search
        
    Returns:
        List of driver keys (lowercase identifiers)
    """
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    mentioned = []
    
    for driver_key, driver_info in config.DRIVERS.items():
        for alias in driver_info['aliases']:
            # Word boundary matching
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                if driver_key not in mentioned:
                    mentioned.append(driver_key)
                break
    
    return mentioned


def detect_teams(text: str) -> List[str]:
    """
    Detect F1 teams mentioned in text.
    
    Args:
        text: Text to search
        
    Returns:
        List of team keys
    """
    if not text or not isinstance(text, str):
        return []
    
    text_lower = text.lower()
    mentioned = []
    
    for team_key, team_info in config.TEAMS.items():
        for alias in team_info['aliases']:
            pattern = r'\b' + re.escape(alias) + r'\b'
            if re.search(pattern, text_lower):
                if team_key not in mentioned:
                    mentioned.append(team_key)
                break
    
    return mentioned


def detect_rivalries(text: str) -> List[Tuple[str, str]]:
    """
    Detect rival driver pairs mentioned together in text.
    
    Args:
        text: Text to search
        
    Returns:
        List of (driver1, driver2) tuples
    """
    mentioned_drivers = set(detect_drivers(text))
    rivalries_found = []
    
    for driver1, driver2 in config.RIVALRIES:
        if driver1 in mentioned_drivers and driver2 in mentioned_drivers:
            rivalries_found.append((driver1, driver2))
    
    return rivalries_found


# =============================================================================
# Text Processing
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean text for NLP processing.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def clean_text_for_wordcloud(text: str) -> str:
    """
    Clean text for word cloud generation (remove more noise).
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text suitable for word cloud
    """
    text = clean_text(text)
    
    # Remove special characters but keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.lower().strip()


def extract_keywords(texts: List[str], top_n: int = 50) -> List[Tuple[str, int]]:
    """
    Extract most frequent keywords from a list of texts.
    
    Args:
        texts: List of text strings
        top_n: Number of top keywords to return
        
    Returns:
        List of (word, count) tuples
    """
    initialize_nltk()
    
    # Combine all texts
    all_text = ' '.join([clean_text_for_wordcloud(t) for t in texts if t])
    
    # Tokenize
    try:
        tokens = word_tokenize(all_text)
    except:
        tokens = all_text.split()
    
    # Get stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    
    # Add F1-specific stopwords
    f1_stopwords = {
        'f1', 'formula', 'one', 'race', 'racing', 'video', 'like', 'would', 
        'get', 'got', 'really', 'think', 'know', 'just', 'dont', "don't",
        'im', "i'm", 'thats', "that's", 'hes', "he's", 'theyre', "they're",
        'going', 'want', 'make', 'even', 'still', 'need', 'see', 'way',
        'much', 'year', 'time', 'good', 'best', 'great', 'could', 'would',
        'also', 'back', 'first', 'last', 'next', 'well', 'come', 'look',
        'say', 'said', 'thing', 'people', 'right', 'take', 'every', 'new',
        'season', 'driver', 'drivers', 'team', 'teams', 'car', 'cars'
    }
    stop_words.update(f1_stopwords)
    
    # Filter tokens
    filtered = [
        token for token in tokens
        if token not in stop_words
        and len(token) > 2
        and token.isalpha()
    ]
    
    return Counter(filtered).most_common(top_n)


def detect_topics(text: str) -> Dict[str, int]:
    """
    Detect F1-specific topics in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of {topic: count}
    """
    if not text or not isinstance(text, str):
        return {}
    
    text_lower = text.lower()
    topic_counts = {}
    
    for topic, keywords in config.F1_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            topic_counts[topic] = count
    
    return topic_counts


# =============================================================================
# Metric Calculations
# =============================================================================

def calculate_engagement_rate(view_count: int, like_count: int, comment_count: int) -> float:
    """
    Calculate engagement rate for a video.
    
    Formula: (Likes + Comments) / Views * 100
    
    Args:
        view_count: Number of views
        like_count: Number of likes
        comment_count: Number of comments
        
    Returns:
        Engagement rate as percentage
    """
    if view_count == 0:
        return 0.0
    return ((like_count + comment_count) / view_count) * 100


def calculate_controversy_index(comment_count: int, like_count: int) -> float:
    """
    Calculate controversy index (comment-to-like ratio).
    
    Higher values indicate more debate/controversy.
    
    Args:
        comment_count: Number of comments
        like_count: Number of likes
        
    Returns:
        Controversy index
    """
    if like_count == 0:
        return 0.0
    return comment_count / like_count


# =============================================================================
# Duration Parsing
# =============================================================================

def parse_duration(duration_str: str) -> int:
    """
    Parse ISO 8601 duration string to seconds.
    
    Args:
        duration_str: Duration string (e.g., 'PT4M13S', 'PT1H30M')
        
    Returns:
        Duration in seconds
    """
    if not duration_str or duration_str == 'PT0S':
        return 0
    
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int) -> str:
    """
    Format seconds as human-readable duration.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., '1:30:45' or '4:30')
    """
    if seconds < 0:
        return '0:00'
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f'{hours}:{minutes:02d}:{secs:02d}'
    return f'{minutes}:{secs:02d}'


# =============================================================================
# Number Formatting
# =============================================================================

def format_number(num: int) -> str:
    """
    Format large numbers for display.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., '1.5M', '45.2K')
    """
    if num >= 1_000_000:
        return f'{num/1_000_000:.1f}M'
    elif num >= 1_000:
        return f'{num/1_000:.1f}K'
    return str(num)


# =============================================================================
# Temporal Feature Extraction
# =============================================================================

def extract_temporal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.
    
    Args:
        df: DataFrame with date column
        date_column: Name of the datetime column
        
    Returns:
        DataFrame with added temporal columns
    """
    df = df.copy()
    
    # Parse datetime
    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    
    # Extract features
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['day_name'] = df[date_column].dt.day_name()
    df['hour'] = df[date_column].dt.hour
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    return df


# =============================================================================
# Data Validation
# =============================================================================

def validate_videos_df(df: pd.DataFrame) -> bool:
    """Check if videos DataFrame has required columns."""
    required = ['video_id', 'title', 'view_count', 'like_count', 'comment_count']
    return all(col in df.columns for col in required)


def validate_comments_df(df: pd.DataFrame) -> bool:
    """Check if comments DataFrame has required columns."""
    required = ['comment_id', 'video_id', 'text_original', 'like_count']
    return all(col in df.columns for col in required)


# =============================================================================
# DataFrame Utilities
# =============================================================================

def safe_json_loads(s: str, default=None):
    """Safely load JSON string, returning default on failure."""
    import json
    try:
        return json.loads(s)
    except:
        return default if default is not None else []


def list_to_string(lst: List) -> str:
    """Convert list to pipe-separated string for CSV storage."""
    if not lst:
        return ''
    return '|'.join(str(item) for item in lst)


def string_to_list(s: str) -> List[str]:
    """Convert pipe-separated string back to list."""
    if not s or pd.isna(s):
        return []
    return [item.strip() for item in s.split('|') if item.strip()]
