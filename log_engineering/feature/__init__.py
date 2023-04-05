from log_engineering.feature import time

TIME_METHODS = {
    "remaining_time": time.remaining_time,
    "execution_time": time.execution_time,
    "accumulated_time": time.accumulated_time,
    "within_day": time.within_day,
    "within_week": time.within_week
}

def available_methods():
    print("Time-based features: ", *TIME_METHODS.keys())

__all__ = ["time"]
