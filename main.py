import argparse
import pandas as pd
from log_engineering.feature import time, TIME_METHODS, available_methods

time_column = "time"
log = pd.read_csv("/home/seidi/Repositores/pm_projects/sampling_pm/data/log.csv")
if log[time_column].dtype == pd.CategoricalDtype:
    log[time_column] = pd.to_datetime(
        log[time_column], infer_datetime_format=True
    ).dt.tz_localize(None)

available_methods()
to_sec = lambda x: x.dt.total_seconds()
log["remaining_time"] = (
    log
    .groupby("case_id")[time_column]
    .transform(time.remaining_time)
    .transform(to_sec)
)
