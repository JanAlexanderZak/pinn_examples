from typing import Tuple, List, Callable

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def shuffle_df(_df: pd.DataFrame, random_seed: int = 1) -> pd.DataFrame:
    """ Shuffle whole pd.df

    Returns:
        pd.DataFrame: Shuffled pd.df.
    """
    return _df.sample(frac=1, random_state=random_seed).reset_index(drop=True)


def get_numeric(_df) -> pd.DataFrame:
    """ Filters for numeric columns.

    Args:
        df (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """
    numeric = ["int16", "int32", "int64", "float16", "float32", "float64"]
    return _df.select_dtypes(include=numeric)


def train_val_test_split(
    df: pd.DataFrame,
    target_columns: List[str],
    ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Formula:
        1.
        Val/Test size = 100 * (1 - 0.6) = 0.4
        <-> Train size = 1 - 0.4 = 0.6
        2.
        Val size = 0.4 * (0.2 / (0.2 + 0.2)) = 0.2
        Val size = 0.4 * (test / (test + val)) = proportion of test on test/val = 0.2
        <-> Test size = 0.4 - 0.2 = 0.2
    """
    x_complete = df.drop(target_columns, axis="columns")
    y_complete = df[target_columns]

    for target_column in target_columns:
        assert target_column not in x_complete
    assert len(x_complete.columns) > len(y_complete.columns)

    x_train, x_val_test, y_train, y_val_test = train_test_split(
        x_complete,
        y_complete,
        test_size=(1 - ratio[0]),
        random_state=7,
        shuffle=True,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test,
        y_val_test,
        test_size=(ratio[2] / (ratio[2] + ratio[1])),
        random_state=7,
        shuffle=True,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def normalize(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    norm_type: str = "z-score",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Callable]:
    """ Min-Max Scaling and z-Score Standardization.

    Args:
        x_values (pd.DataFrame): x values as pd.df.
        norm_type (str, optional): Scaler. Defaults to "z-score".

    Raises:
        ValueError: Raises if pd.Scaler is unknown.

    Returns:
        pd.DataFrame: Normalized df (x_values).
    """
    if norm_type == "min-max":
        scaler = MinMaxScaler()
    elif norm_type == "z-score":
        scaler = StandardScaler()
    else:
        raise ValueError("Flag 'norm_type' is neither 'min-max' nor 'z-score'.")

    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    x_test_norm = scaler.transform(x_test)

    return x_train_norm, x_val_norm, x_test_norm, scaler
