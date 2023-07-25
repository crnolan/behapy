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
                allow_exact_matches: bool = True) -> pd.DataFrame:
    """Find events relative to other events and return their latencies.

    Args:
        events (pd.DataFrame): events DataFrame
        reference (str): event of interest
        source (str): event by which to filter the reference event
        direction (Literal['backward', 'forward']):
            direction of the reference event _from_ the source event
        allow_exact_matches (bool):
            whether to allow exact time matches
    """
    groups = events.groupby(['subject', 'session', 'task', 'run'])
    if len(groups) > 1:
        return groups.apply(find_events,
                            reference=reference,
                            source=source,
                            direction=direction,
                            allow_exact_matches=allow_exact_matches)
    # rdf = events.loc[events.event_id == reference, 'onset'].to_frame()
    # rdf = rdf.set_index('onset', drop=False)
    rdf = events.droplevel(['subject', 'session', 'task', 'run'])
    rdf = rdf.loc[rdf.event_id == reference, :].reset_index()
    rdf = rdf.set_index('onset', drop=False)
    # tdf = events.loc[events.event_id == source, 'onset'].to_frame()
    # tdf = tdf.set_index('onset')
    tdf = pd.DataFrame(index=events.loc[events.event_id == source, :].index)
    tdf = tdf.droplevel(['subject', 'session', 'task', 'run'])
    tdf.index = tdf.index.set_names('source_onset')
    if len(tdf) == 0 or len(rdf) == 0:
        return pd.DataFrame(columns=['onset', 'duration', 'latency']).set_index('onset')
    df = pd.merge_asof(tdf, rdf,
                       left_index=True, right_index=True,
                       direction=direction,
                       allow_exact_matches=allow_exact_matches).dropna().reset_index()
    df['latency'] = df['onset'] - df['source_onset']
    return (df.loc[:, ['duration', 'latency', 'onset']].groupby('onset').min())


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
    for event in events.event_id.unique():
        matrix, offsets = _build_single_regressor(
            data, events.loc[events.event_id == event], window=window)
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

