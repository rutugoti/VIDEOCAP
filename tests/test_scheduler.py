import pytest
import time
from src.orchestration.scheduler import BatchScheduler

def test_scheduler_modes():
    scheduler = BatchScheduler(total_time_limit_sec=100.0, target_margin_sec=10.0)
    scheduler.start_batch(total_videos=5)
    
    # First clip should be BALANCED
    assert scheduler.decide_execution_mode() == "BALANCED"
    
    # Register 1 clip completed quickly (e.g. 5s)
    scheduler.register_completed(elapsed=5.0, api_calls=3, tokens=100)
    
    # Total limit is 90s. Elapsed is 5s. Remaining is 85s.
    # Remaining videos = 4. Avg time = 5s. Projected = 20s.
    # 20s is < 85s * 0.5 (42.5s). So we should upgrade to DEEP.
    assert scheduler.decide_execution_mode() == "DEEP"
    
    # Register 2 clips completed slowly (e.g. 35s each)
    scheduler.register_completed(elapsed=35.0, api_calls=5, tokens=200)
    scheduler.register_completed(elapsed=35.0, api_calls=5, tokens=200)
    
    # Total completed = 3. Remaining videos = 2.
    # Avg time = (5 + 35 + 35)/3 = 25s. Projected = 50s.
    # Elapsed is around 75s. Remaining is 15s.
    # 50s is > 15s * 0.9 (13.5s). So we should demote to FAST.
    scheduler.start_time = time.time() - 75.0  # mock elapsed time
    assert scheduler.decide_execution_mode() == "FAST"

def test_scheduler_retries():
    scheduler = BatchScheduler(total_time_limit_sec=100.0, target_margin_sec=10.0)
    scheduler.start_batch(total_videos=3)
    
    # First clip: no metrics, can_retry is False (avg_time defaults)
    assert scheduler.can_retry("FAST") is False
    assert scheduler.can_retry("BALANCED") is False
    
    # Complete one quickly
    scheduler.register_completed(elapsed=5.0, api_calls=2, tokens=50)
    # Remaining videos = 2. Avg time = 5. Remaining budget = 90 - 5 = 85.
    # can_retry in BALANCED: (85 - 5) > (2 * 5) -> 80 > 10 -> True
    scheduler.start_time = time.time() - 5.0
    assert scheduler.can_retry("BALANCED") is True
    assert scheduler.can_retry("DEEP") is True
    
    # If running behind
    scheduler.start_time = time.time() - 85.0  # 85s elapsed
    # Remaining budget = 90 - 85 = 5. Remaining videos = 2. Avg time = 5.
    # (5 - 5) > (2 * 5) -> 0 > 10 -> False
    assert scheduler.can_retry("BALANCED") is False
    assert scheduler.can_retry("DEEP") is False
