# %%
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from behapy.utils import load_preprocessed_experiment
from behapy.events import build_design_matrix, regress, find_events
import statsmodels.api as sm
import seaborn as sns
sns.set_theme()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
BIDSROOT = Path('..')
pre = load_preprocessed_experiment(BIDSROOT)
dff_id = ['subject', 'session', 'task', 'run', 'label']
dff_recordings = pre.recordings.loc[:, dff_id].drop_duplicates()

# %%
# z-score the dff
dff = pre.dff.copy()
dff['dff'] = dff.dff.groupby(dff_id, group_keys=False).apply(lambda df: (df - df.mean()) / df.std())

# %%
# Map IPSI and CONTRA
def _map_ipsi_contra(row):
    r = row.iloc[0]
    if r.label == 'RDMS':
        events = pre.events.loc[(r.subject, r.session, r.task, r.run)].replace({'rlp': 'ipsilp', 'llp': 'contralp'})
    elif r.label == 'LDMS':
        events = pre.events.loc[(r.subject, r.session, r.task, r.run)].replace({'rlp': 'contralp', 'llp': 'ipsilp'})
    else:
        raise ValueError(f'Unknown label {r.label}')
    return events.sort_index(level='onset')


events = dff_recordings.groupby(dff_id).apply(_map_ipsi_contra)
events = events.droplevel('label')
events

# %%
# Map events to individual recordings
def _get_nonevent(events, sub_events):
    nonevent = events.loc[:, ['duration']].merge(sub_events.loc[:, ['latency']], how='left', left_index=True, right_index=True, indicator=True)
    return nonevent.loc[nonevent._merge == 'left_only', ['duration', 'latency']]


REWmag = find_events(events, 'mag', ['pel', 'suc'])
NOREWmag = _get_nonevent(events.loc[events.event_id == 'mag', :], REWmag)
first_ipsilp = find_events(events, 'ipsilp', ['ipsilp', 'contralp', 'mag'], allow_exact_matches=False)
first_ipsilp = first_ipsilp.loc[first_ipsilp.latency < pd.to_timedelta('2s')]
notfirst_ipsilp = _get_nonevent(events.loc[events.event_id == 'ipsilp', :], first_ipsilp)
first_contralp = find_events(events, 'contralp', ['ipsilp', 'contralp', 'mag'], allow_exact_matches=False)
first_contralp = first_contralp.loc[first_contralp.latency < pd.to_timedelta('2s')]
notfirst_contralp = _get_nonevent(events.loc[events.event_id == 'contralp', :], first_contralp)
new_events = pd.concat([REWmag, NOREWmag, first_ipsilp, notfirst_ipsilp, first_contralp, notfirst_contralp],
                       keys=['REWmag', 'NOREWmag', 'first_ipsilp', 'notfirst_ipsilp', 'first_contralp', 'notfirst_contralp'],
                       names=['event_id'])
new_events = new_events.reset_index('event_id').loc[:, ['duration', 'event_id']]
events = pd.concat([events, new_events]).sort_index()

# %%
plot_meta = {'Magazine': ['REWmag', 'NOREWmag'],
             'Reward': ['pel', 'suc'],
             'First press': ['first_ipsilp', 'first_contralp'],
             'Other press': ['notfirst_ipsilp', 'notfirst_contralp']}
event_ids_of_interest = sum(plot_meta.values(), [])
events_of_interest = events.loc[events.event_id.isin(event_ids_of_interest), :]

# %%
def _build_design_matrix(row):
    r = row.iloc[0]
    return build_design_matrix(
        dff.loc[(r.subject, r.session, r.task, r.run, r.label), :],
        events_of_interest.loc[(r.subject, r.session, r.task, r.run), :],
        (-1, 2))


design_matrix = dff_recordings.groupby(dff_id).apply(_build_design_matrix).fillna(False).astype(bool)

# %%
idx = pd.IndexSlice
dm_filt = design_matrix.loc[idx[:, :, ['FI15', 'RR5', 'RR10'], :, :, :], :].sort_index()


def _regress(df):
    return regress(df, dff.loc[df.index, 'dff'], min_events=25)


# %%
r1 = dm_filt.loc[:, idx[sum(plot_meta.values(), []), :]].groupby(level=('subject', 'task'), group_keys=True).apply(_regress)

# %%
# s1 = r1.stack(0).stack()
s1 = r1.copy()
s1.name = 'beta'
s1 = s1.reset_index()
s1['event_type'] = s1.event.map({v: k for k, l in plot_meta.items() for v in l})
sns.relplot(data=s1, x='offset', y='beta', hue='event', row='event_type',
            col='task',
            kind='line', hue_order=sum(plot_meta.values(), []), aspect=2)

# %%
