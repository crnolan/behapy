# %%

import os
import sys
import numpy as np
import pandas as pd
import tdt

sys.path.insert(0,'../src/behapy')

# %%
from config import getParameters
from fp import normalise
from visUtils import *

p,sinfo = getParameters()


# %%

#for f,fn in enumerate(sinfo):
fn=sinfo[0]
print('\n')

fnl = os.path.join(p['dataDir'],fn)
print('reading data from {}'.format(fnl))

# %%
# format data
block = tdt.read_block(fnl)
data = block.streams._465A.data
fs = block.streams._465A.fs
ts = np.arange(data.shape[0]) / fs
df = pd.Series(block.streams._465A.data, index=ts)
control = pd.Series(block.streams._405A.data,index=ts)
#TODO: make a separate script w functions to convert tdt source data to "raw" numpy arrays in BIDS-friendly format

# %%
# TODO: find discontinuities
# is that discontinuity hitting zero (i.e. is it likely that the fibre was disconnected?) if yes, we want to cut this section
# if disconnected, does it come back on? if yes, we want to mask out disconnect but keep both good bits
# has the baseline for good data shifted? if yes, we *might* want to fit the isosbestic trace in two separate regressions

# %%
# interactively view raw traces 
trace_shade = visRawInteractive(df)

# %%
# TODO: plot the auto-generated "bad" intervals 
# TODO: make sure people can remove these

# %%
# flag artefacts
#TODO: get this into utilities. shouldn't expect people to cp big chunks into their analysis script.
import holoviews as hv
import holoviews.streams as streams
from intervaltree import IntervalTree, Interval
reject_intervals = IntervalTree()
def reject_overlay(intervals, selected=[]):
    if intervals is None or intervals == []:
        return hv.Overlay([])
    colors = ['red' if x in selected else 'pink' for x in range(len(intervals))]
    return hv.Overlay([hv.VSpan(*(interval[0:2])).opts(color=c)
                       for interval, c in zip(intervals, colors)])
def record_reject(boundsx, x, y):
    if None not in [x, y]:
        reject_intervals.remove_overlap(x)
    if boundsx is not None and None not in boundsx:
        reject_intervals.add(Interval(*boundsx))
        reject_intervals.merge_overlaps()
    return reject_overlay(reject_intervals)
reject_stream = streams.BoundsX(source=trace_shade, boundsx=None, transient=True)
select_stream = streams.DoubleTap(source=trace_shade, x=None, y=None, transient=True)
reject_dmap = hv.DynamicMap(record_reject, streams=[reject_stream, select_stream])
(trace_shade * reject_dmap).opts(height=300, responsive=True)

# %%
# save segments marked as artefacts to a .csv
rejected = treeToList(reject_intervals,2)
fnl = os.path.join(p['dataDir'],fn,fn + '_rejected.csv')
np.savetxt(fnl,rejected,delimiter=",")

# %%
# normalise
#   read info about rejected intervals
rejected = np.loadtxt(fnl,delimiter=",")
mask = rejectedToMask(rejected,ts)
#   implement your nomalisation
dff = normalise(df,control,mask,fs,method=p['normMethod'],detrend=p['normDetrend'])


# %%
