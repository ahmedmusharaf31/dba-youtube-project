from main_script import CachedOpenF1DataPipeline, CachedDriverPerformanceAnalyzer, OpenF1DataPipeline

# Test the cache
print("Testing cache system...")
pipeline = CachedOpenF1DataPipeline()
analyzer = CachedDriverPerformanceAnalyzer(pipeline)

# First run (will fetch from API)
print("\n=== FIRST RUN (will fetch from API) ===")
scores1 = analyzer.calculate_composite_score(2024, use_cache=True)

# Second run (should use cache)
print("\n=== SECOND RUN (should use cache) ===")
scores2 = analyzer.calculate_composite_score(2024, use_cache=True)

print(f"\nFirst run got {len(scores1) if scores1 else 0} drivers")
print(f"Second run got {len(scores2) if scores2 else 0} drivers")

# Check if they're the same
if scores1 and scores2:
    print(f"Cache working: {len(scores1) == len(scores2)}")
