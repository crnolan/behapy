# %%
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from behapy.utils import load_preprocessed_experiment
from behapy.events import build_design_matrix, regress, find_events
import statsmodels.api as sm
import seaborn as sns
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
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
        events = pre.events.loc[(r.subject, r.session, r.task, r.run, r.label)].replace({'rlp': 'ipsilp', 'llp': 'contralp'})
    elif r.label == 'LDMS':
        events = pre.events.loc[(r.subject, r.session, r.task, r.run, r.label)].replace({'rlp': 'contralp', 'llp': 'ipsilp'})
    else:
        raise ValueError(f'Unknown label {r.label}')
    return events.sort_index(level='onset')


events = dff_recordings.groupby(dff_id).apply(_map_ipsi_contra)
events

# %%
# Map events to individual recordings
def _get_nonevent(events, sub_events):
    nonevent = events.loc[:, ['duration']].merge(sub_events.loc[:, ['latency']], how='left', left_index=True, right_index=True, indicator=True)
    return nonevent.loc[nonevent._merge == 'left_only', ['duration', 'latency']]


REWmag = find_events(events, 'mag', ['pel', 'suc'])
NOREWmag = _get_nonevent(events.loc[events.event_id == 'mag', :], REWmag)
first_ipsilp = find_events(events, 'ipsilp', ['ipsilp', 'contralp', 'mag'], allow_exact_matches=False)
# first_ipsilp = first_ipsilp.loc[first_ipsilp.latency < pd.to_timedelta('2s')]
first_ipsilp = first_ipsilp.loc[first_ipsilp.latency < 2]
notfirst_ipsilp = _get_nonevent(events.loc[events.event_id == 'ipsilp', :], first_ipsilp)
first_contralp = find_events(events, 'contralp', ['ipsilp', 'contralp', 'mag'], allow_exact_matches=False)
# first_contralp = first_contralp.loc[first_contralp.latency < pd.to_timedelta('2s')]
first_contralp = first_contralp.loc[first_contralp.latency < 2]
notfirst_contralp = _get_nonevent(events.loc[events.event_id == 'contralp', :], first_contralp)
new_events = pd.concat([REWmag, NOREWmag, first_ipsilp, notfirst_ipsilp, first_contralp, notfirst_contralp],
                       keys=['REWmag', 'NOREWmag', 'first_ipsilp', 'notfirst_ipsilp', 'first_contralp', 'notfirst_contralp'],
                       names=['event_id'])
new_events = new_events.reset_index('event_id').loc[:, ['duration', 'event_id']]
events = pd.concat([events, new_events]).sort_index()

# %%
plot_meta = {'Magazine': ['REWmag', 'NOREWmag']}
# plot_meta = {'Magazine': ['REWmag', 'NOREWmag'],
#              'Reward': ['pel', 'suc'],
#              'First press': ['first_ipsilp', 'first_contralp'],
#              'Other press': ['notfirst_ipsilp', 'notfirst_contralp']}
event_ids_of_interest = sum(plot_meta.values(), [])
events_of_interest = events.loc[events.event_id.isin(event_ids_of_interest), :]

# %%
def _build_design_matrix(row):
    r = row.iloc[0]
    return build_design_matrix(
        dff.loc[(r.subject, r.session, r.task, r.run, r.label), :],
        events_of_interest.loc[(r.subject, r.session, r.task, r.run, r.label), :],
        (-1, 2))


design_matrix = dff_recordings.groupby(dff_id).apply(_build_design_matrix).fillna(False).astype(bool)

# %%
idx = pd.IndexSlice
dm_filt = design_matrix.loc[idx[:, :, ['FI15', 'RR5', 'RR10'], :, :, :], :].sort_index()


def _regress(df):
    return regress(df, dff.loc[df.index, 'dff'], min_events=25)

# %%
# activate R magic
pandas2ri.activate()

# # path to R
# r_file_path = '..' # add path
# # open the file and read it into a string
# with open(r_file_path, 'r') as file:
#     r_script_string = file.read()

# code temporarily stored in this string, move to file later
r_code = '''
suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(parallel))
suppressPackageStartupMessages(library(cAIC4))
suppressPackageStartupMessages(library(magrittr))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(MASS))
suppressPackageStartupMessages(library(lsei))
suppressPackageStartupMessages(library(refund))
suppressPackageStartupMessages(library(stringr))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(mvtnorm))
suppressPackageStartupMessages(library(arrangements))
suppressPackageStartupMessages(library(progress))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(gridExtra))
suppressPackageStartupMessages(library(Rfast))
suppressPackageStartupMessages(library(fastFMM))

dat <- read.csv("/Users/uqdkilpa/Documents/Code/Clones/fastFMM/vignettes/time_series.csv")

mod <- fui(Y ~ treatment + # main effect of cue
              (treatment | id),  # random slope & intercept
              data = dat,
              parallel = TRUE,
              analytic = FALSE) # bootstrap

mod_qn <- mod$qn
mod_resid <- mod$residuals
mod_bootsamps <- mod$bootstrap_samps
mod_argvals <- mod$argvals
mod_aic <- mod$aic
mod_betahat <- mod$betaHat
mod_betahatvar <- mod$betaHat.var
'''

# execute the string in R
robjects.r(r_code)

# collect aic/bic 
mod_aic = pd.DataFrame(robjects.r['mod_aic'])
mod_aic.columns = ['AIC', 'BIC', 'cAIC']

# collect qn 
mod_qn = robjects.r['mod_qn'] 

# collect bootstrap samples
mod_bootsamps = robjects.r['mod_bootsamps']

# collect argument values
mod_argvals = np.array(robjects.r['mod_argvals'])

# collect coefficients
mod_betaHat = pd.DataFrame(robjects.r['mod_betahat'])

# collect coefficient variances
mod_betahat_var = np.array(robjects.r['mod_betahatvar'])

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
