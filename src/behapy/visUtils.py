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
def visRawInteractive(df):
    return datashade(hv.Curve((df.index, df)), aggregator=ds.count()).redim(
    x='time', y=hv.Dimension('F')).opts(width=800, tools=['xbox_select, tap'])

def getMask(trace_shade):
    reject_stream = streams.BoundsX(source=trace_shade, boundsx=None, transient=True)
    select_stream = streams.DoubleTap(source=trace_shade, x=None, y=None, transient=True)
    reject_dmap = hv.DynamicMap(record_reject, streams=[reject_stream, select_stream])
    (trace_shade * reject_dmap).opts(height=300, responsive=True)

def treeToList(tree,max):
    mylist=[]
    for i in tree:
        mylist.append(list(i[0:max]))
    return mylist

def rejectedToMask(rejected,ts):

    mask = np.array(np.zeros((ts.shape)),dtype="bool")
    if np.any(rejected):
        if len(np.shape(rejected))==1:
            rejected=[rejected]
        for i in range(0,np.shape(rejected)[0]):
            min = np.where(ts>rejected[i][0])[0][0]
            max = np.where(ts<rejected[i][1])[0][-1]
            mask[range(min,max)]=True

    return mask