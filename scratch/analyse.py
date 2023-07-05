# %%
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import behapy.fp as fp
from behapy.events import load_events, build_design_matrix, regress
from behapy.pathutils import list_preprocessed
import statsmodels.api as sm
import seaborn as sns
sns.set_theme()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
BIDSROOT = Path('/scratch/cnolan/TB006')
recordings = pd.DataFrame(list_preprocessed(BIDSROOT))


# %%
def _load_preprocessed(row):
    r = row.iloc[0]
    return fp.load_preprocessed(BIDSROOT, r.subject, r.session,
                                r.task, r.run, r.label)


def _load_events(row):
    r = row.iloc[0]
    return load_events(BIDSROOT, r.subject, r.session, r.task, r.run)


dff_id = ['subject', 'session', 'task', 'run', 'label']
dff_recordings = recordings.loc[:, dff_id].drop_duplicates()
dff = dff_recordings.groupby(dff_id).apply(_load_preprocessed)
dff.attrs = _load_preprocessed(recordings.iloc[[0]]).attrs
events_id = ['subject', 'session', 'task', 'run']
events_recordings = recordings.loc[:, events_id].drop_duplicates()
events = events_recordings.groupby(events_id).apply(_load_events)
events.attrs = _load_events(recordings.iloc[[0]]).attrs

# %%
def _build_design_matrix(row):
    r = row.iloc[0]
    return build_design_matrix(
        dff.loc[(r.subject, r.session, r.task, r.run, r.label), :],
        events.loc[(r.subject, r.session, r.task, r.run), :],
        (-1, 2))


design_matrix = dff_recordings.groupby(dff_id).apply(_build_design_matrix).fillna(False).astype(bool)

# %%
idx = pd.IndexSlice
fi15_design = design_matrix.loc[idx[:, :, 'FI15', :, :, :], :].sort_index()


def _regress(df):
    return regress(df, dff.loc[df.index, 'dff'], min_events=25)


# %%
plot_meta = {'Magazine': ['mag'],
             'Reward': ['pel', 'suc'],
             'Lever press': ['rlp', 'llp']}
r1 = fi15_design.loc[:, idx[sum(plot_meta.values(), []), :]].groupby(level=('subject'), group_keys=True).apply(_regress)

# %%
s1 = r1.stack(0).stack()
s1.name = 'beta'
s1 = s1.reset_index()
s1['event_type'] = s1.event.map({v: k for k, l in plot_meta.items() for v in l})
sns.relplot(data=s1, x='offset', y='beta', hue='event', row='event_type',
            kind='line', hue_order=sum(plot_meta.values(), []))
# %%
