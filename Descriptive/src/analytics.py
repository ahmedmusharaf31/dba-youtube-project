"""
Descriptive Analytics Module for F1 YouTube Data

Computes all metrics:
1. Driver Engagement Analytics (SoV, Sentiment, Headline Impact, Rivalry)
2. Audience Sentiment & Reaction (Global Sentiment, Top Comments, Controversy)
3. Video Performance Metrics (Engagement Rate, Reach, Virality)
4. Content & Topic Analysis (Keywords, Team Mentions, Temporal Activity)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from . import config
from . import utils


class F1DescriptiveAnalytics:
    """
    Compute descriptive analytics metrics for F1 YouTube data.
    """
    
    def __init__(self, videos_df: pd.DataFrame, comments_df: pd.DataFrame):
        """
        Initialize with video and comment data.
        
        Args:
            videos_df: DataFrame with video data
            comments_df: DataFrame with comment data
        """
        self.videos = videos_df.copy()
        self.comments = comments_df.copy()
        
        # Preprocess data
        self._preprocess()
    
    def _preprocess(self):
        """Preprocess dataframes for analysis."""
        # Parse dates
        if 'published_at' in self.videos.columns:
            self.videos = utils.extract_temporal_features(self.videos, 'published_at')
        
        if 'published_at' in self.comments.columns:
            self.comments = utils.extract_temporal_features(self.comments, 'published_at')
        
        # Parse video duration
        if 'duration' in self.videos.columns:
            self.videos['duration_seconds'] = self.videos['duration'].apply(utils.parse_duration)
        
        # Calculate engagement rate for videos
        if all(col in self.videos.columns for col in ['view_count', 'like_count', 'comment_count']):
            self.videos['engagement_rate'] = self.videos.apply(
                lambda row: utils.calculate_engagement_rate(
                    row['view_count'], row['like_count'], row['comment_count']
                ), axis=1
            )
            
            self.videos['controversy_index'] = self.videos.apply(
                lambda row: utils.calculate_controversy_index(
                    row['comment_count'], row['like_count']
                ), axis=1
            )
        
        # Detect drivers and teams in comments
        if 'text_original' in self.comments.columns:
            self.comments['drivers_mentioned'] = self.comments['text_original'].apply(
                lambda x: utils.detect_drivers(x) if isinstance(x, str) else []
            )
            self.comments['teams_mentioned'] = self.comments['text_original'].apply(
                lambda x: utils.detect_teams(x) if isinstance(x, str) else []
            )
            self.comments['rivalries_detected'] = self.comments['text_original'].apply(
                lambda x: utils.detect_rivalries(x) if isinstance(x, str) else []
            )
        
        # Detect drivers in video titles
        if 'title' in self.videos.columns:
            self.videos['drivers_in_title'] = self.videos['title'].apply(
                lambda x: utils.detect_drivers(x) if isinstance(x, str) else []
            )
    
    # =========================================================================
    # 1. DRIVER ENGAGEMENT ANALYTICS
    # =========================================================================
    
    def driver_share_of_voice(self) -> pd.DataFrame:
        """
        Calculate Driver Share of Voice (SoV).
        
        Returns percentage of comments mentioning each driver.
        """
        driver_counts = Counter()
        total_comments = len(self.comments)
        
        for drivers in self.comments['drivers_mentioned']:
            for driver in drivers:
                driver_counts[driver] += 1
        
        # Comments mentioning at least one driver
        comments_with_drivers = sum(1 for d in self.comments['drivers_mentioned'] if len(d) > 0)
        
        results = []
        for driver, count in driver_counts.most_common():
            results.append({
                'driver': driver,
                'mention_count': count,
                'share_of_voice_pct': (count / total_comments * 100) if total_comments > 0 else 0,
                'share_of_driver_comments_pct': (count / comments_with_drivers * 100) if comments_with_drivers > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    def driver_sentiment_scores(self) -> pd.DataFrame:
        """
        Calculate average sentiment for comments mentioning each driver.
        """
        # First compute sentiment for all comments
        if 'sentiment_compound' not in self.comments.columns:
            print("Computing sentiment for comments...")
            sentiments = utils.batch_sentiment_analysis(
                self.comments['text_original'].fillna('').tolist()
            )
            self.comments['sentiment_compound'] = sentiments['compound']
            self.comments['sentiment_label'] = sentiments['label']
        
        driver_sentiments = {}
        
        for idx, row in self.comments.iterrows():
            for driver in row['drivers_mentioned']:
                if driver not in driver_sentiments:
                    driver_sentiments[driver] = []
                driver_sentiments[driver].append(row['sentiment_compound'])
        
        results = []
        for driver, scores in driver_sentiments.items():
            scores_arr = np.array(scores)
            results.append({
                'driver': driver,
                'avg_sentiment': np.mean(scores_arr),
                'std_sentiment': np.std(scores_arr),
                'positive_pct': (scores_arr > 0.05).mean() * 100,
                'negative_pct': (scores_arr < -0.05).mean() * 100,
                'neutral_pct': ((scores_arr >= -0.05) & (scores_arr <= 0.05)).mean() * 100,
                'comment_count': len(scores)
            })
        
        return pd.DataFrame(results).sort_values('avg_sentiment', ascending=False)
    
    def headline_impact(self) -> pd.DataFrame:
        """
        Calculate average performance metrics for videos featuring each driver in title.
        """
        driver_video_stats = {}
        
        for idx, row in self.videos.iterrows():
            for driver in row.get('drivers_in_title', []):
                if driver not in driver_video_stats:
                    driver_video_stats[driver] = {
                        'view_counts': [],
                        'like_counts': [],
                        'comment_counts': [],
                        'engagement_rates': []
                    }
                driver_video_stats[driver]['view_counts'].append(row.get('view_count', 0))
                driver_video_stats[driver]['like_counts'].append(row.get('like_count', 0))
                driver_video_stats[driver]['comment_counts'].append(row.get('comment_count', 0))
                driver_video_stats[driver]['engagement_rates'].append(row.get('engagement_rate', 0))
        
        results = []
        for driver, stats in driver_video_stats.items():
            results.append({
                'driver': driver,
                'video_count': len(stats['view_counts']),
                'avg_views': np.mean(stats['view_counts']),
                'avg_likes': np.mean(stats['like_counts']),
                'avg_comments': np.mean(stats['comment_counts']),
                'avg_engagement_rate': np.mean(stats['engagement_rates']),
                'total_views': np.sum(stats['view_counts'])
            })
        
        return pd.DataFrame(results).sort_values('avg_views', ascending=False)
    
    def rivalry_intensity(self) -> pd.DataFrame:
        """
        Count comments mentioning rival driver pairs.
        """
        rivalry_counts = Counter()
        
        for rivalries in self.comments['rivalries_detected']:
            for rivalry in rivalries:
                # Sort to ensure consistent key
                key = tuple(sorted(rivalry))
                rivalry_counts[key] += 1
        
        results = []
        for rivalry, count in rivalry_counts.most_common():
            results.append({
                'driver_1': rivalry[0],
                'driver_2': rivalry[1],
                'rivalry_pair': f"{rivalry[0]} vs {rivalry[1]}",
                'comment_count': count,
                'intensity_pct': (count / len(self.comments) * 100) if len(self.comments) > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    # =========================================================================
    # 2. AUDIENCE SENTIMENT & REACTION
    # =========================================================================
    
    def global_sentiment_distribution(self) -> Dict[str, Any]:
        """
        Calculate overall sentiment distribution across all comments.
        """
        # Ensure sentiment is computed
        if 'sentiment_compound' not in self.comments.columns:
            sentiments = utils.batch_sentiment_analysis(
                self.comments['text_original'].fillna('').tolist()
            )
            self.comments['sentiment_compound'] = sentiments['compound']
            self.comments['sentiment_label'] = sentiments['label']
        
        total = len(self.comments)
        distribution = self.comments['sentiment_label'].value_counts()
        
        return {
            'total_comments': total,
            'positive_count': distribution.get('positive', 0),
            'neutral_count': distribution.get('neutral', 0),
            'negative_count': distribution.get('negative', 0),
            'positive_pct': distribution.get('positive', 0) / total * 100 if total > 0 else 0,
            'neutral_pct': distribution.get('neutral', 0) / total * 100 if total > 0 else 0,
            'negative_pct': distribution.get('negative', 0) / total * 100 if total > 0 else 0,
            'avg_sentiment': self.comments['sentiment_compound'].mean(),
            'std_sentiment': self.comments['sentiment_compound'].std(),
        }
    
    def top_fan_favorite_comments(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top comments by like count.
        """
        if 'like_count' not in self.comments.columns:
            return pd.DataFrame()
        
        top_comments = self.comments.nlargest(top_n, 'like_count')[
            ['comment_id', 'video_id', 'author_display_name', 'text_original',
             'like_count', 'published_at']
        ].copy()
        
        # Add video title
        if 'video_id' in top_comments.columns and 'video_id' in self.videos.columns:
            video_titles = self.videos.set_index('video_id')['title'].to_dict()
            top_comments['video_title'] = top_comments['video_id'].map(video_titles)
        
        return top_comments
    
    def controversy_index_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        Rank videos by controversy index (comment-to-like ratio).
        """
        if 'controversy_index' not in self.videos.columns:
            return pd.DataFrame()
        
        # Filter out infinite values and get top controversial
        controversial = self.videos[
            (self.videos['controversy_index'] != float('inf')) &
            (self.videos['controversy_index'] > 0)
        ].nlargest(top_n, 'controversy_index')[
            ['video_id', 'title', 'view_count', 'like_count', 'comment_count',
             'controversy_index', 'engagement_rate']
        ]
        
        return controversial
    
    def polarity_vs_performance_correlation(self) -> Dict[str, float]:
        """
        Calculate correlation between video sentiment and performance metrics.
        """
        # Compute average sentiment per video from comments
        if 'sentiment_compound' not in self.comments.columns:
            sentiments = utils.batch_sentiment_analysis(
                self.comments['text_original'].fillna('').tolist()
            )
            self.comments['sentiment_compound'] = sentiments['compound']
        
        video_sentiment = self.comments.groupby('video_id')['sentiment_compound'].mean().reset_index()
        video_sentiment.columns = ['video_id', 'avg_comment_sentiment']
        
        # Merge with video metrics
        merged = self.videos.merge(video_sentiment, on='video_id', how='inner')
        
        if len(merged) < 5:
            return {'error': 'Not enough data for correlation analysis'}
        
        correlations = {}
        for metric in ['view_count', 'like_count', 'comment_count', 'engagement_rate']:
            if metric in merged.columns:
                corr = merged['avg_comment_sentiment'].corr(merged[metric])
                correlations[f'sentiment_vs_{metric}'] = corr
        
        return correlations
    
    # =========================================================================
    # 3. VIDEO PERFORMANCE METRICS
    # =========================================================================
    
    def engagement_rate_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        Rank videos by engagement rate.
        """
        return self.videos.nlargest(top_n, 'engagement_rate')[
            ['video_id', 'title', 'view_count', 'like_count', 'comment_count',
             'engagement_rate', 'published_at']
        ]
    
    def video_reach_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        Rank videos by view count.
        """
        return self.videos.nlargest(top_n, 'view_count')[
            ['video_id', 'title', 'view_count', 'like_count', 'comment_count',
             'engagement_rate', 'published_at']
        ]
    
    def virality_potential_ranking(self, top_n: int = 20) -> pd.DataFrame:
        """
        Rank videos by comment count (immediate user response).
        """
        return self.videos.nlargest(top_n, 'comment_count')[
            ['video_id', 'title', 'view_count', 'like_count', 'comment_count',
             'engagement_rate', 'published_at']
        ]
    
    def video_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall video performance statistics.
        """
        return {
            'total_videos': len(self.videos),
            'total_views': self.videos['view_count'].sum(),
            'total_likes': self.videos['like_count'].sum(),
            'total_comments': self.videos['comment_count'].sum(),
            'avg_views': self.videos['view_count'].mean(),
            'avg_likes': self.videos['like_count'].mean(),
            'avg_comments': self.videos['comment_count'].mean(),
            'avg_engagement_rate': self.videos['engagement_rate'].mean(),
            'median_views': self.videos['view_count'].median(),
            'max_views': self.videos['view_count'].max(),
        }
    
    # =========================================================================
    # 4. CONTENT & TOPIC ANALYSIS
    # =========================================================================
    
    def dominant_keywords(self, source: str = 'both', top_n: int = 50) -> List[Tuple[str, int]]:
        """
        Extract dominant keywords from content.
        
        Args:
            source: 'comments', 'descriptions', or 'both'
            top_n: Number of top keywords to return
        """
        texts = []
        
        if source in ['comments', 'both']:
            if 'text_original' in self.comments.columns:
                texts.extend(self.comments['text_original'].fillna('').tolist())
        
        if source in ['descriptions', 'both']:
            # Some workflows (e.g., loading a minimal videos table from processed reports)
            # may not include a `description` column.
            if 'description' in self.videos.columns:
                texts.extend(self.videos['description'].fillna('').tolist())
            elif 'title' in self.videos.columns:
                # Fallback: use titles as a proxy for descriptions so keyword extraction still works.
                texts.extend(self.videos['title'].fillna('').tolist())
        
        return utils.extract_keywords(texts, top_n=top_n)
    
    def team_mention_frequency(self) -> pd.DataFrame:
        """
        Count mentions of each F1 team in comments.
        """
        team_counts = Counter()
        
        for teams in self.comments['teams_mentioned']:
            for team in teams:
                team_counts[team] += 1
        
        results = []
        total = len(self.comments)
        for team, count in team_counts.most_common():
            results.append({
                'team': team,
                'mention_count': count,
                'mention_pct': (count / total * 100) if total > 0 else 0,
                'color': config.TEAM_COLORS.get(team, '#808080')
            })
        
        return pd.DataFrame(results)
    
    def temporal_activity_videos(self) -> pd.DataFrame:
        """
        Analyze when videos are uploaded (day of week, hour).
        """
        if 'day_of_week' not in self.videos.columns:
            return pd.DataFrame()
        
        # Create heatmap data: day of week vs hour
        pivot = self.videos.groupby(['day_name', 'hour']).size().reset_index(name='count')
        
        # Also get summary by day
        by_day = self.videos.groupby('day_name').agg({
            'video_id': 'count',
            'view_count': 'mean'
        }).reset_index()
        by_day.columns = ['day_name', 'video_count', 'avg_views']
        
        return by_day
    
    def temporal_activity_comments(self) -> pd.DataFrame:
        """
        Analyze when comments are posted (day of week, hour).
        """
        if 'day_of_week' not in self.comments.columns:
            return pd.DataFrame()
        
        by_day = self.comments.groupby('day_name').size().reset_index(name='comment_count')
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        by_day['day_name'] = pd.Categorical(by_day['day_name'], categories=day_order, ordered=True)
        by_day = by_day.sort_values('day_name')
        
        return by_day
    
    def temporal_heatmap_data(self) -> pd.DataFrame:
        """
        Create heatmap data for comment activity by day and hour.
        """
        if 'day_of_week' not in self.comments.columns or 'hour' not in self.comments.columns:
            return pd.DataFrame()
        
        heatmap = self.comments.groupby(['day_name', 'hour']).size().reset_index(name='count')
        
        # Pivot for heatmap
        pivot = heatmap.pivot(index='day_name', columns='hour', values='count').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        return pivot
    
    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================
    
    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analytics report.
        """
        print("Generating comprehensive analytics report...")
        
        report = {
            # 1. Driver Engagement
            'driver_share_of_voice': self.driver_share_of_voice(),
            'driver_sentiment_scores': self.driver_sentiment_scores(),
            'headline_impact': self.headline_impact(),
            'rivalry_intensity': self.rivalry_intensity(),
            
            # 2. Audience Sentiment
            'global_sentiment': self.global_sentiment_distribution(),
            'top_comments': self.top_fan_favorite_comments(10),
            'controversial_videos': self.controversy_index_ranking(20),
            'polarity_performance_correlation': self.polarity_vs_performance_correlation(),
            
            # 3. Video Performance
            'top_engagement': self.engagement_rate_ranking(20),
            'top_reach': self.video_reach_ranking(20),
            'top_virality': self.virality_potential_ranking(20),
            'performance_summary': self.video_performance_summary(),
            
            # 4. Content Analysis
            'top_keywords': self.dominant_keywords('both', 50),
            'team_mentions': self.team_mention_frequency(),
            'video_temporal': self.temporal_activity_videos(),
            'comment_temporal': self.temporal_activity_comments(),
        }
        
        print("Report generation complete!")
        return report
    
    def save_report_to_csv(self, output_dir: Optional[str] = None):
        """
        Save all analytics DataFrames to CSV files.
        """
        output_dir = output_dir or config.PROCESSED_DATA_DIR
        output_path = config.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_full_report()
        
        # Save DataFrames
        dataframes = {
            'driver_share_of_voice': report['driver_share_of_voice'],
            'driver_sentiment_scores': report['driver_sentiment_scores'],
            'headline_impact': report['headline_impact'],
            'rivalry_intensity': report['rivalry_intensity'],
            'top_comments': report['top_comments'],
            'controversial_videos': report['controversial_videos'],
            'top_engagement': report['top_engagement'],
            'top_reach': report['top_reach'],
            'top_virality': report['top_virality'],
            'team_mentions': report['team_mentions'],
            'video_temporal': report['video_temporal'],
            'comment_temporal': report['comment_temporal'],
        }
        
        for name, df in dataframes.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                filepath = output_path / f'{name}.csv'
                df.to_csv(filepath, index=False)
                print(f"Saved: {filepath}")
        
        # Save summary statistics as JSON
        import json
        summary = {
            'global_sentiment': report['global_sentiment'],
            'polarity_performance_correlation': report['polarity_performance_correlation'],
            'performance_summary': report['performance_summary'],
            'top_keywords': report['top_keywords'][:30],  # Top 30 for JSON
        }
        
        summary_path = output_path / 'analytics_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved: {summary_path}")
        
        # Save processed comments with sentiment
        if 'sentiment_compound' in self.comments.columns:
            self.comments.to_csv(output_path / 'comments_with_sentiment.csv', index=False)
            print(f"Saved: {output_path / 'comments_with_sentiment.csv'}")
