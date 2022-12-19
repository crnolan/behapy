# %%

from vis_utils import *
from fp import normalise
from utils import get_parameters
import os
import sys
import numpy as np
import pandas as pd
import tdt

sys.path.insert(0, '../src/behapy')

# %%

p = get_parameters()

# %%

# for f,fn in enumerate(p['sinfo']):
fn = p['sinfo'][0]
print('\n')

fnl = os.path.join(p['dataDir'], fn)
print('reading data from {}'.format(fnl))

# %%
# format data
block = tdt.read_block(fnl)
data = block.streams._465A.data
fs = block.streams._465A.fs
ts = np.arange(data.shape[0]) / fs
df = pd.Series(block.streams._465A.data, index=ts)
control = pd.Series(block.streams._405A.data, index=ts)
# TODO: make a separate script w functions to convert tdt source data to
# "raw" numpy arrays in BIDS-friendly format

# %%
# interactively view raw traces
fig = vis_raw_interactive(df, detectBads=True, detectShifts=False)
fig

# %%
# save segments marked as artefacts to a .csv
rejected = treeToList(reject_intervals, 2)
fnl = os.path.join(p['dataDir'], fn, fn + '_rejected.csv')
np.savetxt(fnl, rejected, delimiter=",")

# %%
# normalise
#   read info about rejected intervals
rejected = np.loadtxt(fnl, delimiter=",")
mask = rejectedToMask(rejected, ts)
#   implement your nomalisation
dff = normalise(
    df,
    control,
    mask,
    fs,
    method=p['normMethod'],
    detrend=p['normDetrend'])


# %%
