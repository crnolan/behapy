from typing import Tuple, Iterable, Literal
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


def find_events(events_df: pd.DataFrame,
                reference: str,
                source: str,
                direction: Literal['backward', 'forward'] = 'forward',
                allow_exact_matches: bool = True) -> pd.Series:
    rdf = events_df.loc[events_df.value == reference, 'onset'].to_frame()
    rdf = rdf.set_index('onset', drop=False)
    tdf = events_df.loc[events_df.value == source, 'onset'].to_frame()
    tdf = tdf.set_index('onset')
    tdf.index.name = 'source_onset'
    df = pd.merge_asof(tdf, rdf,
                       left_index=True, right_index=True,
                       direction=direction,
                       allow_exact_matches=allow_exact_matches).dropna().reset_index()
    df['latency'] = df['onset'] - df['source_onset']
    return (df.loc[:, ['latency', 'onset']].groupby('onset').min())['latency']

