from typing import Tuple, Iterable
from pathlib import Path
import json
import logging
import numpy as np
from collections import namedtuple
from typing import Iterable
import pandas as pd
import statsmodels.api as sm
from .pathutils import get_events_path


def load_events(root: Path,
                subject: str,
                session: str,
                task: str,
                run: str) -> pd.DataFrame:
    """Load events from a BIDS root directory.

    Args:
        root (Path): path to the root of the BIDS dataset
        subject (str): subject ID
        session (str): session ID
        task (str): task ID
        run (str): run ID
    """
    events_path = get_events_path(root, subject, session, task, run)
    if not events_path.exists():
        raise ValueError(f'Events file {events_path} does not exist')
    events = pd.read_csv(events_path, index_col=0)
    return events


# def find_events_within(events_df: pd.DataFrame,
#                        reference: str,
#                        target: str,
#                        direction: str = 'forward',
#                        tmin: float = 0.,
#                        tmax: float = np.inf) -> pd.Series:
#     ref = pd.Series(index=events_df.value == reference
#     # Inputs need to be Series with timestamps
#     event1_mask.name = 'src'
#     event2_mask.name = 'dest'
#     df1 = pd.DataFrame(index=event1_mask.index[event1_mask])
#     df1.index.name = 'onset'
#     # Subtract or add one microsecond to ensure the same event isn't
#     # detected as following or preceding itself (if the same event train
#     # is passed in)
#     if allow_simultaneous:
#         offset = 0
#     else:
#         offset = -1e-6 if direction == 'forward' else 1e-6
#     df2 = pd.DataFrame(event2_mask.index[event2_mask].to_numpy(),
#                        index=(event2_mask.index[event2_mask] + offset),
#                        columns=['indices'])
#     merged = pd.merge_asof(df1, df2,
#                            left_index=True, right_index=True,
#                            direction=direction).reset_index()
#     merged = merged[(merged.indices - merged.onset).abs() < tmax]
#     series = pd.Series([False]*len(event2_mask), index=event2_mask.index)
#     series.loc[merged.indices[~merged.indices.isna()]] = True
#     return series.to_numpy()