# %%
import numpy as np
import pandas as pd
from tdt import read_block
from pathlib import Path
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread, rasterize
hv.extension('bokeh')
import panel
panel.extension(comms='vscode')

idx = pd.IndexSlice

# %%
block_path = Path('../sourcedata/Rats18-21-210330-104533')
block = read_block(str(block_path))
data = block.streams._465A.data
fs = block.streams._465A.fs
ts = np.arange(data.shape[0]) / fs
df = pd.Series(block.streams._465A.data, index=ts)

# %%
datashade(hv.Curve((df.index, df)), aggregator=ds.count()).redim(
    x='time', y=hv.Dimension('F')).opts(width=800)

# %%
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


# %%
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
#     if boundsx is None:
#         return hv.Overlay([])
    if None not in [x, y]:
        reject_intervals.remove_overlap(x)
    if boundsx is not None and None not in boundsx:
        reject_intervals.add(Interval(*boundsx))
        reject_intervals.merge_overlaps()
    return reject_overlay(reject_intervals)

trace_shade = datashade(hv.Curve((df.index, df)), 
    aggregator=ds.count()).redim(x='time', y=hv.Dimension('F')).opts(width=800, tools=['xbox_select, tap'])

reject_stream = streams.BoundsX(source=trace_shade, boundsx=None, transient=True)
select_stream = streams.DoubleTap(source=trace_shade, x=None, y=None, transient=True)
reject_dmap = hv.DynamicMap(record_reject, streams=[reject_stream, select_stream])
(trace_shade * reject_dmap).opts(height=300, responsive=True)

# %%
