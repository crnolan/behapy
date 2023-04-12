from functools import partial
from intervaltree import Interval, IntervalTree
import pandas as pd
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread, rasterize
hv.extension('bokeh')
import holoviews.streams as streams
import panel as pn
import param
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
pn.extension('tabulator', comms='vscode')


def interval_overlay(intervals, selected=[]):
    if intervals is None or intervals == []:
        return hv.Overlay([])
    colors = ['red' if x in selected else 'pink' for x in range(len(intervals))]
    return hv.Overlay([hv.VSpan(*(interval[0:2])).opts(color=c)
                       for interval, c in zip(intervals, colors)])


def record_intervals(boundsx, x, y, intervals):
#     if boundsx is None:
#         return hv.Overlay([])
    if None not in [x, y]:
        intervals.remove_overlap(x)
    if boundsx is not None and None not in boundsx:
        intervals.add(Interval(*boundsx))
        intervals.merge_overlaps()
    return interval_overlay(intervals)


def interval_overlay_map(trace, intervals):
    interval_stream = streams.BoundsX(source=trace, boundsx=None,
                                      transient=True)
    select_stream = streams.DoubleTap(source=trace, x=None, y=None,
                                      transient=True)
    return hv.DynamicMap(partial(record_intervals, intervals=intervals),
                         streams=[interval_stream, select_stream])
    

class PreprocessDashboard(param.Parameterized):
    selected_index = param.Integer(default=None, allow_None=True)
    metadata_table = param.DataFrame(
        pd.DataFrame(columns=['subject', 'session', 'label']))

    def __init__(self, recordings, data_func, regress_func, **params):
        super().__init__(**params)
        self.metadata_table = recordings
        self.data_func = data_func
        self.regress_func = regress_func
        self.recording = None
        self.intervals = None
        
    def on_selection_change(self, event):
        if event.new:
            self.selected_index = event.new[0]
            self.recording, intervals = self.data_func(self.selected_index)
            # Convert invervals from indexes to timestamps
            intervals = [(self.recording.ts[i], self.recording.ts[j])
                          for i, j in intervals]
            # Convert the intervals to an IntervalTree
            self.intervals = IntervalTree.from_tuples(intervals)
        else:
            self.selected_index = None

    def create_tabulator_widget(self):
        # Create a Tabulator widget with the metadata_table DataFrame
        tabulator_widget = pn.widgets.Tabulator(self.metadata_table)

        # Attach the callback function to the 'selection' parameter of
        # the Tabulator widget
        tabulator_widget.param.watch(self.on_selection_change, 'selection')

        return tabulator_widget
    
    @param.depends("selected_index")
    def plot_rejection(self):
        if self.recording is None:
            return
        iso_shade = datashade(hv.Curve((self.recording.ts, self.recording.iso())),
                              aggregator=ds.count(), cmap='blue')
        iso_shade = iso_shade.opts(default_tools=[], tools=['xbox_select, xwheel_zoom'])
        overlay = interval_overlay_map(iso_shade, self.intervals)
        # Add horizontal selection tool
        plot = (iso_shade * overlay).opts(responsive=True, min_width=600, min_height=200)
        return plot

    @param.depends("selected_index", "intervals")
    def plot_regression(self):
        if self.recording is None:
            return
        # Calculate a robust regression of each signal vs. the
        # isosbestic channel and plot
        regression = self.regress_func(self.recording)
        reg_shade = datashade(hv.Curve((self.recording.ts, regression.fittedvalues)),
                              aggregator=ds.count(), cmap='green')
        iso_shade = datashade(hv.Curve((self.recording.ts, self.recording.iso())),
                              aggregator=ds.count(), cmap='red')
        sig_shade = datashade(hv.Curve((self.recording.ts, self.recording.signal())),
                              aggregator=ds.count(), cmap='blue')
        plot = (iso_shade * sig_shade * reg_shade).opts(responsive=True, min_width=600, min_height=300)
        return plot

    def view(self):
        return pn.Column(
            pn.Row(
                self.create_tabulator_widget(),
                self.plot_rejection),
            pn.Row(
                self.plot_regression)
            )