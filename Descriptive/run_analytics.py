"""
Main script to run F1 YouTube Descriptive Analytics Pipeline

Usage:
    python run_analytics.py                    # Use existing data
    python run_analytics.py --extract          # Extract new data first
    python run_analytics.py --extract --max-videos 100 --max-comments 50
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src.youtube_extractor import YouTubeF1Extractor, load_existing_data
from src.analytics import F1DescriptiveAnalytics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='F1 YouTube Descriptive Analytics Pipeline'
    )
    parser.add_argument(
        '--extract', '-e',
        action='store_true',
        help='Extract fresh data from YouTube API (requires API key)'
    )
    parser.add_argument(
        '--max-videos',
        type=int,
        default=config.MAX_VIDEOS,
        help=f'Maximum videos to fetch (default: {config.MAX_VIDEOS})'
    )
    parser.add_argument(
        '--max-comments',
        type=int,
        default=config.MAX_COMMENTS_PER_VIDEO,
        help=f'Maximum comments per video (default: {config.MAX_COMMENTS_PER_VIDEO})'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=config.SEASON_START_DATE,
        help=f'Start date in ISO format (default: {config.SEASON_START_DATE[:10]})'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=config.SEASON_END_DATE,
        help=f'End date in ISO format (default: {config.SEASON_END_DATE[:10]})'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='YouTube API key (or set YOUTUBE_API_KEY env variable)'
    )
    
    return parser.parse_args()


def print_banner():
    """Print application banner."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║          F1 YouTube Descriptive Analytics Pipeline            ║
║                                                               ║
║  Analyzing Formula 1's YouTube presence:                      ║
║  • Driver Engagement & Share of Voice                         ║
║  • Audience Sentiment Analysis                                ║
║  • Video Performance Metrics                                  ║
║  • Content & Topic Analysis                                   ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def print_sentiment_summary(sentiment: dict):
    """Print global sentiment summary."""
    print("\n📊 GLOBAL SENTIMENT DISTRIBUTION")
    print("-" * 40)
    print(f"  Total Comments Analyzed: {sentiment['total_comments']:,}")
    print(f"  Average Sentiment Score: {sentiment['avg_sentiment']:.3f}")
    print()
<<<<<<< HEAD
    print(f"  😊 Positive: {sentiment['positive_count']:,} ({sentiment['positive_pct']:.1f}%)")
    print(f"  😐 Neutral:  {sentiment['neutral_count']:,} ({sentiment['neutral_pct']:.1f}%)")
    print(f"  😞 Negative: {sentiment['negative_count']:,} ({sentiment['negative_pct']:.1f}%)")
=======
    print(f"  Positive: {sentiment['positive_count']:,} ({sentiment['positive_pct']:.1f}%)")
    print(f"  Neutral:  {sentiment['neutral_count']:,} ({sentiment['neutral_pct']:.1f}%)")
    print(f"  Negative: {sentiment['negative_count']:,} ({sentiment['negative_pct']:.1f}%)")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb


def print_driver_sov(sov_df):
    """Print driver share of voice summary."""
<<<<<<< HEAD
    print("\n🏎️  DRIVER SHARE OF VOICE (Top 10)")
=======
    print("\nDRIVER SHARE OF VOICE (Top 10)")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print("-" * 50)
    print(f"{'Driver':<15} {'Mentions':>10} {'SoV %':>10}")
    print("-" * 50)
    for _, row in sov_df.head(10).iterrows():
        print(f"{row['driver']:<15} {row['mention_count']:>10,} {row['share_of_voice_pct']:>9.1f}%")


def print_driver_sentiment(sentiment_df):
    """Print driver sentiment summary."""
<<<<<<< HEAD
    print("\n💭 DRIVER SENTIMENT SCORES (sorted by avg sentiment)")
=======
    print("\nDRIVER SENTIMENT SCORES (sorted by avg sentiment)")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print("-" * 70)
    print(f"{'Driver':<15} {'Avg Sent':>10} {'Positive%':>10} {'Negative%':>10} {'Comments':>10}")
    print("-" * 70)
    for _, row in sentiment_df.head(10).iterrows():
<<<<<<< HEAD
        emoji = "😊" if row['avg_sentiment'] > 0.1 else ("😞" if row['avg_sentiment'] < -0.1 else "😐")
=======
        emoji = "Positive" if row['avg_sentiment'] > 0.1 else ("Negative" if row['avg_sentiment'] < -0.1 else "Neutral")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
        print(f"{emoji} {row['driver']:<13} {row['avg_sentiment']:>10.3f} {row['positive_pct']:>9.1f}% {row['negative_pct']:>9.1f}% {row['comment_count']:>10,}")


def print_rivalry_intensity(rivalry_df):
    """Print rivalry intensity summary."""
<<<<<<< HEAD
    print("\n⚔️  RIVALRY INTENSITY")
=======
    print("\nRIVALRY INTENSITY")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print("-" * 50)
    if rivalry_df.empty:
        print("  No rivalry mentions detected in comments.")
    else:
        for _, row in rivalry_df.head(5).iterrows():
            print(f"  {row['rivalry_pair']}: {row['comment_count']:,} comments ({row['intensity_pct']:.2f}%)")


def print_video_performance(summary: dict):
    """Print video performance summary."""
<<<<<<< HEAD
    print("\n📹 VIDEO PERFORMANCE SUMMARY")
=======
    print("\nVIDEO PERFORMANCE SUMMARY")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print("-" * 40)
    print(f"  Total Videos: {summary['total_videos']:,}")
    print(f"  Total Views: {summary['total_views']:,}")
    print(f"  Total Likes: {summary['total_likes']:,}")
    print(f"  Total Comments: {summary['total_comments']:,}")
    print()
    print(f"  Avg Views per Video: {summary['avg_views']:,.0f}")
    print(f"  Avg Engagement Rate: {summary['avg_engagement_rate']:.2f}%")
    print(f"  Max Views (single video): {summary['max_views']:,}")


def print_team_mentions(team_df):
    """Print team mention frequency."""
<<<<<<< HEAD
    print("\n🏆 TEAM MENTION FREQUENCY")
=======
    print("\nTEAM MENTION FREQUENCY")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print("-" * 50)
    for _, row in team_df.iterrows():
        bar_length = int(row['mention_pct'] * 2)
        bar = "█" * bar_length
        print(f"  {row['team']:<15} {bar} {row['mention_count']:,} ({row['mention_pct']:.1f}%)")


def print_top_keywords(keywords):
    """Print top keywords."""
<<<<<<< HEAD
    print("\n🔤 TOP KEYWORDS")
=======
    print("\nTOP KEYWORDS")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print("-" * 40)
    print("  " + ", ".join([f"{word} ({count})" for word, count in keywords[:15]]))


def main():
    """Main entry point."""
    print_banner()
    args = parse_args()
    
    # Step 1: Load or extract data
    if args.extract:
        print("=" * 60)
        print("STEP 1: Extracting data from YouTube API...")
        print("=" * 60)
        
        api_key = args.api_key or config.YOUTUBE_API_KEY
        if not api_key:
            print("ERROR: No API key provided. Set YOUTUBE_API_KEY or use --api-key")
            sys.exit(1)
        
        extractor = YouTubeF1Extractor(api_key=api_key)
        
        # Ensure date format
        start_date = args.start_date if 'T' in args.start_date else f"{args.start_date}T00:00:00Z"
        end_date = args.end_date if 'T' in args.end_date else f"{args.end_date}T23:59:59Z"
        
        videos_df, comments_df = extractor.extract_all(
            max_videos=args.max_videos,
            max_comments_per_video=args.max_comments,
            published_after=start_date,
            published_before=end_date
        )
    else:
        print("=" * 60)
        print("STEP 1: Loading existing data from CSV files...")
        print("=" * 60)
        
        videos_df, comments_df = load_existing_data()
        
        if videos_df.empty or comments_df.empty:
            print("\nNo existing data found. Run with --extract flag to fetch data.")
            print("Example: python run_analytics.py --extract --max-videos 100")
            sys.exit(1)
    
<<<<<<< HEAD
    print(f"\n✓ Loaded {len(videos_df)} videos and {len(comments_df)} comments")
=======
    print(f"\nLoaded {len(videos_df)} videos and {len(comments_df)} comments")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    
    # Step 2: Run analytics
    print("\n" + "=" * 60)
    print("STEP 2: Running Descriptive Analytics...")
    print("=" * 60)
    
    analytics = F1DescriptiveAnalytics(videos_df, comments_df)
    report = analytics.generate_full_report()
    
    # Step 3: Print summary results
    print("\n" + "=" * 60)
    print("STEP 3: Analytics Results Summary")
    print("=" * 60)
    
    # Print formatted summaries
    print_sentiment_summary(report['global_sentiment'])
    print_driver_sov(report['driver_share_of_voice'])
    print_driver_sentiment(report['driver_sentiment_scores'])
    print_rivalry_intensity(report['rivalry_intensity'])
    print_video_performance(report['performance_summary'])
    print_team_mentions(report['team_mentions'])
    print_top_keywords(report['top_keywords'])
    
    # Step 4: Save results
    print("\n" + "=" * 60)
    print("STEP 4: Saving Results to CSV...")
    print("=" * 60)
    
    analytics.save_report_to_csv()
    
    print("\n" + "=" * 60)
<<<<<<< HEAD
    print("✓ Pipeline Complete!")
=======
    print("Pipeline Complete!")
>>>>>>> 10231d243c4568e5b36e60521c9c081ca25932eb
    print(f"  Results saved to: {config.PROCESSED_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
