import pandas as pd
from typing import Union
from datetime import timedelta


def execution_time(x: Union[pd.Series, pd.core.groupby.DataFrameGroupBy]):
    """
    returns
    Args:
        x (ndarray, pd.Series): array of timestamp values

    Returns:
        _type_: _description_
    """
    from pandas import NaT

    x = x - x.shift(1, fill_value=NaT)
    return x


def accumulated_time(x: Union[pd.Series, pd.core.groupby.DataFrameGroupBy]):
    """
    returns x_{i} - x_{0} given an array of dt values
    Args:
        x (ndarray, pd.Series): array of timestamp values

    Returns:
        _type_: _description_
    """
    x = x - x.min()
    return x


def remaining_time(x: Union[pd.Series, pd.core.groupby.DataFrameGroupBy]):
    """Returns the remaining time of a case

    Args:
        x (_type_): _description_
    """
    x = x.max() - x
    return x


def within_day(x: Union[pd.Series, pd.core.groupby.DataFrameGroupBy]):
    """Returns the time within the day
    x_i - midnight
    Args:
        x (_type_): _description_
    """
    x = x - x.replace(hour=0, minute=0, second=0, microsecond=0)
    return x


def within_week(x: Union[pd.Series, pd.core.groupby.DataFrameGroupBy]):
    """Returns the time within the week w.r.t last sunday

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """

    ls = x - timedelta(x.day_of_week + 1) if x.day_of_week != 6 else x
    ls = ls.replace(hour=0, minute=0, second=0, microsecond=0)
    return x - ls
