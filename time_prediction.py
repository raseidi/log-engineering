import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from log_engineering.utils import read_engineered_data

# from tpot import TPOTRegressor

dataset = "PrepaidTravelCost"
dimensions = 8
file_path = f"data/{dataset}_dim={dimensions}.csv"
data = read_engineered_data(file_path)
data["remaining_time"] = data["remaining_time"].apply(np.log1p)
# data = pd.concat((data, pd.get_dummies(data["concept:name"])), axis=1)
# data.isna().sum()

drop_columns = [
    "concept:name",
    "case:concept:name",
    "time:timestamp",
    "event_id",
    "split_set",
]

df_train = data.loc[data["split_set"] == "train", data.columns.difference(drop_columns)]
df_test = data.loc[data["split_set"] == "test", data.columns.difference(drop_columns)]

regr = RandomForestRegressor(n_jobs=-1)
regr.fit(df_train.drop(["remaining_time"], axis=1), df_train["remaining_time"])
preds = regr.predict(df_test.drop(["remaining_time"], axis=1))
preds /= (24*24*60)
df_test["remaining_time"] /= (24*24*60)
print("MAE:", mean_absolute_error(df_test["remaining_time"], preds))
print("MSE:", mean_squared_error(df_test["remaining_time"], preds))

lstm_mae = 0.0002183
rf_mae = mean_absolute_error(df_test["remaining_time"], preds)
print(lstm_mae / rf_mae)


# tpot = TPOTRegressor(verbosity=2, config_dict="TPOT light", n_jobs=-1)
# tpot.fit(
#     df_train.drop(["case:concept:name", "remaining_time"], axis=1), df_train["remaining_time"]
# )
# preds = tpot.predict(df_test.drop(["case:concept:name", "remaining_time"], axis=1))
# print("MAE:", mean_absolute_error(df_test["remaining_time"], preds))
# print("MSE:", mean_squared_error(df_test["remaining_time"], preds))
