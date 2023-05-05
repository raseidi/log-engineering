from log_engineering.feature import time

TIME_METHODS = {
    "remaining_time": {"fn": time.remaining_time, "groupbycase": True},
    "execution_time": {"fn": time.execution_time, "groupbycase": True},
    "accumulated_time": {"fn": time.accumulated_time, "groupbycase": True},
    "within_day": {"fn": time.within_day, "groupbycase": False},
    "within_week": {"fn": time.within_week, "groupbycase": False},
}


def available_methods():
    return TIME_METHODS.keys()


__all__ = ["time"]
