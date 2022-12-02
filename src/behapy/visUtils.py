'''
lydia barnes 
lydiabarnes@tuta.io

created december 2022
from crnolan visual masking example script

utilities to identify artefacts in and visualise continuous raw fibre photometry data

'''

import numpy as np

import holoviews as hv
import holoviews.streams as streams
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread, rasterize
hv.extension('bokeh')
import panel
panel.extension(comms='vscode')

import datashader as ds
import panel
panel.extension(comms='vscode')
from intervaltree import IntervalTree, Interval
reject_intervals = IntervalTree()

# basic plot, not interactive
def visRaw(df):
    return datashade(hv.Curve((df.index, df)), aggregator=ds.count()).redim(
        x='time', y=hv.Dimension('F')).opts(width=800)

# utilities to support interactive artefact rejection
def get_masked_regions(ts, mask):
    mask_diff = np.concatenate([mask.iloc[:1].astype(int),
                                np.diff(mask.astype(int))])
    mask_onset = ts[mask_diff == 1]
    mask_offset = ts[mask_diff == -1]
    if mask_offset.shape[0] < mask_onset.shape[0]:
        mask_offset = np.concatenate([mask_offset, ts.iloc[-1:]])
    return mask_onset, mask_offset

def plot_regions(onsets, offsets):
    return hv.Overlay([hv.VSpan(t0, t1) for t0, t1 in zip(onsets, offsets)])

def maskToInterval(mask):
    started = False
    sstart=[]
    sstop = []
    for s in range(0,len(mask)):
        if (mask[s]==True) and (started==False):
            sstart.append(np.where(mask[s]))
            started=True
        if (mask[s]==False) and (started==True):
            sstop.append(np.where(mask[s]))
            started=False

    #TODO: turn interval starts and stops into an intervaltree
    return interval

def getAutoBads(df):
    bounds=None

    #find near-zero values (fibre disconnections)
    window=np.round(len(df)*.01)
    mask = np.array(np.zeros((df.shape)),dtype="bool")
    for s in range(0,len(df)-window):
        x = df[s*window:s*window+window]
        if np.median(x)<np.std(df): #if this window is within 1 std of zero
            mask[s+np.round(window*.5)] = True #mask its central value
    #TODO: account for start and end of trace samples that may be bad

    # find jumps
    jumps = np.where(np.abs(np.diff(df))>np.std(np.diff(df))*5)
    print(jumps)
    print(np.shape(jumps))

    #mask jump samples
    for j in range(0,len(jumps)):
        print(jumps[j])
        mask[jumps[j]]==True

    #create intervals from mask


    return bounds

def getAutoShifts(df):
    bounds = None
    # TODO: has the baseline shifted? if yes, we *might* want to fit the isosbestic trace in two separate regressions
    return bounds

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

# interactive plot for artefact rejection
def visRawInteractive(df,detectBads,detectShifts):

    trace_shade = datashade(hv.Curve((df.index, df)), aggregator=ds.count()).redim(
    x='time', y=hv.Dimension('F')).opts(width=800, tools=['xbox_select, tap'])

    bounds=None
    if detectBads:
        bounds = getAutoBads(df) #auto-detect jumps in data
    if detectShifts:
        bounds = getAutoShifts(df) #auto-detect baseline shifts
    reject_stream = streams.BoundsX(source=trace_shade, boundsx=bounds,transient=True) #plot those data segments

    select_stream = streams.DoubleTap(source=trace_shade, x=None, y=None, transient=True)
    reject_dmap = hv.DynamicMap(record_reject, streams=[reject_stream, select_stream])
    
    return (trace_shade * reject_dmap).opts(height=300, responsive=True) #combine trace and dynamic map into one figure, and return it


def treeToList(tree,max):
    mylist=[]
    for i in tree:
        mylist.append(list(i[0:max]))
    return mylist

def intervalToMask(interval,ts):

    mask = np.array(np.zeros((ts.shape)),dtype="bool")
    if np.any(interval):
        if len(np.shape(interval))==1:
            interval=[interval]
        for i in range(0,np.shape(interval)[0]):
            min = np.where(ts>interval[i][0])[0][0]
            max = np.where(ts<interval[i][1])[0][-1]
            mask[range(min,max)]=True

    return mask