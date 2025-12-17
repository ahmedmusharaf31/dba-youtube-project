"""
YouTube Data API v3 Extractor for Formula 1 Channel
Fetches videos and comments for descriptive analytics.
"""
import os
import time
import pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Optional, List, Dict, Any

from . import config


class YouTubeF1Extractor:
    """
    Extract videos and comments from the official F1 YouTube channel
    using the YouTube Data API v3.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor with API credentials.
        
        Args:
            api_key: YouTube Data API key. If None, uses config/env variable.
        """
        self.api_key = api_key or config.YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError(
                "YouTube API key is required. Set YOUTUBE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        self.channel_id = config.F1_CHANNEL_ID
        
        # Ensure data directories exist
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get basic information about the F1 channel."""
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=self.channel_id
            )
            response = request.execute()
            
            if response["items"]:
                channel = response["items"][0]
                return {
                    "channel_id": channel["id"],
                    "title": channel["snippet"]["title"],
                    "description": channel["snippet"].get("description", ""),
                    "subscriber_count": int(channel["statistics"].get("subscriberCount", 0)),
                    "video_count": int(channel["statistics"].get("videoCount", 0)),
                    "view_count": int(channel["statistics"].get("viewCount", 0)),
                    "uploads_playlist_id": channel["contentDetails"]["relatedPlaylists"]["uploads"]
                }
            return {}
        except HttpError as e:
            print(f"Error fetching channel info: {e}")
            return {}
    
    def get_videos(
        self,
        max_results: Optional[int] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        save_to_csv: bool = True
    ) -> pd.DataFrame:
        """
        Fetch videos from the F1 channel.
        
        Args:
            max_results: Maximum number of videos to fetch (None for all)
            published_after: ISO 8601 date string for start date filter
            published_before: ISO 8601 date string for end date filter
            save_to_csv: Whether to save results to CSV
            
        Returns:
            DataFrame with video data
        """
        max_results = max_results or config.MAX_VIDEOS
        published_after = published_after or config.SEASON_START_DATE
        published_before = published_before or config.SEASON_END_DATE
        
        videos = []
        next_page_token = None
        fetched_count = 0
        
        print(f"Fetching videos from F1 channel...")
        print(f"  Date range: {published_after[:10]} to {published_before[:10]}")
        print(f"  Max videos: {max_results}")
        
        while True:
            try:
                # Search for videos on the channel
                request = self.youtube.search().list(
                    part="id,snippet",
                    channelId=self.channel_id,
                    type="video",
                    order="date",
                    publishedAfter=published_after,
                    publishedBefore=published_before,
                    maxResults=min(50, max_results - fetched_count) if max_results else 50,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
                
                if video_ids:
                    # Get detailed statistics for each video
                    stats_request = self.youtube.videos().list(
                        part="snippet,statistics,contentDetails",
                        id=",".join(video_ids)
                    )
                    stats_response = stats_request.execute()
                    
                    for item in stats_response.get("items", []):
                        video_data = self._parse_video(item)
                        videos.append(video_data)
                        fetched_count += 1
                
                print(f"  Fetched {fetched_count} videos...")
                
                # Check if we've reached the limit or no more pages
                next_page_token = response.get("nextPageToken")
                if not next_page_token or (max_results and fetched_count >= max_results):
                    break
                    
                # Rate limiting
                time.sleep(0.1)
                
            except HttpError as e:
                print(f"Error fetching videos: {e}")
                if "quotaExceeded" in str(e):
                    print("API quota exceeded. Try again tomorrow or use a different API key.")
                break
        
        df = pd.DataFrame(videos)
        
        if save_to_csv and not df.empty:
            df.to_csv(config.VIDEOS_CSV, index=False, encoding="utf-8")
            print(f"  Saved {len(df)} videos to {config.VIDEOS_CSV}")
        
        return df
    
    def _parse_video(self, item: Dict) -> Dict[str, Any]:
        """Parse a video item from the API response."""
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})
        
        return {
            "video_id": item["id"],
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "published_at": snippet.get("publishedAt", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "tags": "|".join(snippet.get("tags", [])),
            "category_id": snippet.get("categoryId", ""),
            "duration": content_details.get("duration", ""),
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "favorite_count": int(statistics.get("favoriteCount", 0)),
            "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
        }
    
    def get_comments_for_video(
        self,
        video_id: str,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch top-level comments for a specific video.
        
        Args:
            video_id: YouTube video ID
            max_results: Maximum comments to fetch per video
            
        Returns:
            List of comment dictionaries
        """
        max_results = max_results or config.MAX_COMMENTS_PER_VIDEO
        comments = []
        next_page_token = None
        
        while len(comments) < max_results:
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    order="relevance",
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get("items", []):
                    comment_data = self._parse_comment(item, video_id)
                    comments.append(comment_data)
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
                time.sleep(0.05)  # Rate limiting
                
            except HttpError as e:
                if "commentsDisabled" in str(e):
                    pass  # Comments are disabled for this video
                elif "quotaExceeded" in str(e):
                    print("API quota exceeded.")
                    break
                else:
                    print(f"Error fetching comments for {video_id}: {e}")
                break
        
        return comments
    
    def _parse_comment(self, item: Dict, video_id: str) -> Dict[str, Any]:
        """Parse a comment thread item from the API response."""
        snippet = item.get("snippet", {})
        top_comment = snippet.get("topLevelComment", {}).get("snippet", {})
        
        return {
            "comment_id": item["id"],
            "video_id": video_id,
            "author_display_name": top_comment.get("authorDisplayName", ""),
            "author_channel_id": top_comment.get("authorChannelId", {}).get("value", ""),
            "text_original": top_comment.get("textOriginal", ""),
            "text_display": top_comment.get("textDisplay", ""),
            "like_count": int(top_comment.get("likeCount", 0)),
            "published_at": top_comment.get("publishedAt", ""),
            "updated_at": top_comment.get("updatedAt", ""),
            "reply_count": int(snippet.get("totalReplyCount", 0)),
        }
    
    def get_all_comments(
        self,
        videos_df: Optional[pd.DataFrame] = None,
        max_comments_per_video: Optional[int] = None,
        save_to_csv: bool = True
    ) -> pd.DataFrame:
        """
        Fetch comments for all videos in the dataset.
        
        Args:
            videos_df: DataFrame of videos. If None, loads from CSV.
            max_comments_per_video: Max comments per video
            save_to_csv: Whether to save results to CSV
            
        Returns:
            DataFrame with all comments
        """
        if videos_df is None:
            if config.VIDEOS_CSV.exists():
                videos_df = pd.read_csv(config.VIDEOS_CSV)
            else:
                print("No videos data found. Run get_videos() first.")
                return pd.DataFrame()
        
        max_comments = max_comments_per_video or config.MAX_COMMENTS_PER_VIDEO
        all_comments = []
        total_videos = len(videos_df)
        
        print(f"Fetching comments for {total_videos} videos...")
        
        for idx, row in videos_df.iterrows():
            video_id = row["video_id"]
            video_title = row["title"][:50]
            
            comments = self.get_comments_for_video(video_id, max_comments)
            all_comments.extend(comments)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{total_videos} videos, {len(all_comments)} comments total...")
            
            time.sleep(0.1)  # Rate limiting between videos
        
        df = pd.DataFrame(all_comments)
        
        if save_to_csv and not df.empty:
            df.to_csv(config.COMMENTS_CSV, index=False, encoding="utf-8")
            print(f"  Saved {len(df)} comments to {config.COMMENTS_CSV}")
        
        return df
    
    def extract_all(
        self,
        max_videos: Optional[int] = None,
        max_comments_per_video: Optional[int] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete extraction pipeline: fetch videos and their comments.
        
        Returns:
            Tuple of (videos_df, comments_df)
        """
        print("=" * 60)
        print("F1 YouTube Data Extraction Pipeline")
        print("=" * 60)
        
        # Step 1: Get channel info
        channel_info = self.get_channel_info()
        if channel_info:
            print(f"\nChannel: {channel_info.get('title')}")
            print(f"  Subscribers: {channel_info.get('subscriber_count'):,}")
            print(f"  Total Videos: {channel_info.get('video_count'):,}")
            print(f"  Total Views: {channel_info.get('view_count'):,}")
        
        # Step 2: Fetch videos
        print("\n" + "-" * 40)
        videos_df = self.get_videos(
            max_results=max_videos,
            published_after=published_after,
            published_before=published_before
        )
        
        if videos_df.empty:
            print("No videos found. Check API key and date range.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Step 3: Fetch comments
        print("\n" + "-" * 40)
        comments_df = self.get_all_comments(
            videos_df=videos_df,
            max_comments_per_video=max_comments_per_video
        )
        
        print("\n" + "=" * 60)
        print("Extraction Complete!")
        print(f"  Videos: {len(videos_df)}")
        print(f"  Comments: {len(comments_df)}")
        print("=" * 60)
        
        return videos_df, comments_df


def load_existing_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load existing extracted data from CSV files.
    
    Returns:
        Tuple of (videos_df, comments_df)
    """
    videos_df = pd.DataFrame()
    comments_df = pd.DataFrame()
    
    if config.VIDEOS_CSV.exists():
        videos_df = pd.read_csv(config.VIDEOS_CSV)
        print(f"Loaded {len(videos_df)} videos from {config.VIDEOS_CSV}")
    
    if config.COMMENTS_CSV.exists():
        comments_df = pd.read_csv(config.COMMENTS_CSV)
        print(f"Loaded {len(comments_df)} comments from {config.COMMENTS_CSV}")
    
    return videos_df, comments_df


if __name__ == "__main__":
    # Quick test
    extractor = YouTubeF1Extractor()
    
    # Test channel info
    info = extractor.get_channel_info()
    print(f"Channel: {info.get('title', 'Not found')}")
    
    # Extract a small sample
    videos_df, comments_df = extractor.extract_all(
        max_videos=10,
        max_comments_per_video=20
    )
