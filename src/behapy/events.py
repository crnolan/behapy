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
    events.index = pd.to_timedelta(events.index, unit='s')
    return events


def find_events(events: pd.DataFrame,
                reference: str,
                source: str,
                direction: Literal['backward', 'forward'] = 'forward',
                allow_exact_matches: bool = True) -> pd.Series:
    rdf = events.loc[events.value == reference, 'onset'].to_frame()
    rdf = rdf.set_index('onset', drop=False)
    tdf = events.loc[events.value == source, 'onset'].to_frame()
    tdf = tdf.set_index('onset')
    tdf.index.name = 'source_onset'
    df = pd.merge_asof(tdf, rdf,
                       left_index=True, right_index=True,
                       direction=direction,
                       allow_exact_matches=allow_exact_matches).dropna().reset_index()
    df['latency'] = df['onset'] - df['source_onset']
    return (df.loc[:, ['latency', 'onset']].groupby('onset').min())['latency']


def _find_nearest(origin, fit):
    df0 = pd.DataFrame(np.array(origin), index=origin, columns=['origin'])
    df1 = pd.DataFrame(np.array(fit), index=fit, columns=['fit'])
    first = pd.merge_asof(df1, df0,
                          left_index=True, right_index=True,
                          direction='nearest')
    second = pd.DataFrame(index=origin, dtype=bool)
    second['nearest'] = False
    second.loc[first['origin'], 'nearest'] = True
    return second['nearest'].to_numpy()


def _build_single_regressor(data: pd.DataFrame,
                            events: pd.Series,
                            window: Tuple[float, float]) -> \
                             Tuple[np.ndarray, np.ndarray]:
    event_mask = np.array(_find_nearest(data.index, events.index))
    window_indices = np.round(np.array(window) * data.attrs['fs']).astype(int)
    offsets = np.arange(*window_indices)
    matrix = np.zeros((data.shape[0], offsets.shape[0]))
    for i, offset in enumerate(offsets):
        if offset < 0:
            matrix[:offset, i] = event_mask[-offset:]
        elif offset > 0:
            matrix[offset:, i] = event_mask[:-offset]
        else:
            matrix[:, i] = event_mask
    return matrix, offsets


def build_design_matrix(data: pd.DataFrame,
                        events: pd.DataFrame,
                        window: Tuple[float, float]) -> pd.DataFrame:
    regressor_dfs = []
    for event in events.value.unique():
        matrix, offsets = _build_single_regressor(
            data, events.loc[events.value == event], window=window)
        column_index = pd.MultiIndex.from_product(
            [[event], offsets/data.attrs['fs']], names=('event', 'offset'))
        _df = pd.DataFrame(matrix, dtype=bool, index=data.index,
                           columns=column_index)
        regressor_dfs.append(_df)
    df = pd.concat(regressor_dfs, axis=1)
    df = df.loc[data['mask'], :]
    return df.loc[df.sum(axis=1) > 0, :]


def regress(design_matrix: pd.DataFrame,
            data: pd.DataFrame,
            min_events=50) -> pd.Series:
    dm = design_matrix.loc[:, design_matrix.sum() > min_events]
    if dm.empty:
        return pd.Series(dtype=float, index=dm.columns)
    print(data.shape, dm.shape)
    lr = sm.OLS(data.to_numpy(), dm.to_numpy()).fit()
    return pd.Series(lr.params, index=dm.columns)

