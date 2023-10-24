import logging
from functools import partial
from intervaltree import Interval
import json
from . import fp
from .config import load_preprocess_config
import pandas as pd
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade
import holoviews.streams as streams
import panel as pn
import param
hv.extension('bokeh')
pn.extension('tabulator', comms='vscode')
# pn.extension('tabulator')


def signal_curve(df, y_dim):
    return hv.Curve((df.index.to_numpy(), df), 'time', y_dim)
    # return hv.Curve((pd.to_timedelta(df.index, unit='s'), df))


def signal_shade(df, y_dim, cmap):
    return datashade(signal_curve(df, y_dim=y_dim),
                     aggregator=ds.count(), cmap=cmap)


def interval_overlay(intervals, selected=[]):
    if intervals is None or intervals == []:
        return hv.Overlay([])
    colors = ['red' if x in selected else 'pink' for x in range(len(intervals))]
    return hv.Overlay([hv.VSpan(*(interval[0:2])).opts(color=c)
                       for interval, c in zip(intervals, colors)])


def record_intervals(bounds, x, y, intervals, interval_callback=None):
    if None not in [x, y]:
        intervals.remove_overlap(x)
        logging.debug(f'Intervals now {intervals}')
        if interval_callback is not None:
            interval_callback()
    if bounds is not None and None not in [bounds[0], bounds[2]]:
        intervals.add(Interval(bounds[0], bounds[2]))
        intervals.merge_overlaps()
        logging.debug(f'Intervals now {intervals}')
        if interval_callback is not None:
            interval_callback()
    return interval_overlay(intervals)


def interval_overlay_map(trace, intervals, interval_callback=None):
    interval_stream = streams.BoundsXY(source=trace,
                                       transient=False)
    select_stream = streams.DoubleTap(source=trace, x=None, y=None,
                                      transient=False)
    return hv.DynamicMap(partial(record_intervals, intervals=intervals,
                                 interval_callback=interval_callback),
                         streams=[interval_stream, select_stream])


def rejection_shade(recording, intervals, interval_callback=None, y_dim='raw'):
    isoch = recording.attrs['iso_channel']
    iso_shade = datashade(signal_curve(recording[isoch], y_dim=y_dim),
                          aggregator=ds.count(), cmap='blue')
    iso_shade = iso_shade.opts(default_tools=[], tools=['xbox_select', 'xwheel_zoom'])
    overlay = interval_overlay_map(iso_shade, intervals, interval_callback)
    # Add horizontal selection tool
    plot = (iso_shade * overlay)
    return plot


class PreprocessDashboard(param.Parameterized):
    selected_index = param.Integer(default=None, allow_None=True)
    metadata_table = param.DataFrame(
        pd.DataFrame(columns=['subject', 'session', 'task', 'run', 'label']))
    interval_update = param.Integer(default=0, allow_None=False)
    regression_update = param.Integer(default=0, allow_None=False)

    def __init__(self, recordings, data_func, bidsroot, **params):
        super().__init__(**params)
        self.metadata_table = recordings
        self.data_func = data_func
        self.bidsroot = bidsroot
        self.recording = None
        self.intervals = None
        self.regression = None
        self.dff = None

    def on_selection_change(self, event):
        if event.new:
            selected_index = event.new[0]
            signal = self.data_func(selected_index)
            self.recording = fp.downsample(signal, 64)
            # Check whether there is an interval file
            sa = signal.attrs
            self.intervals = fp.load_rejections(sa['root'], sa['subject'],
                                                sa['session'], sa['task'],
                                                sa['run'], sa['label'])
            if self.intervals is None:
                self.intervals = fp.find_disconnects(self.recording)
                self.update_intervals()
            self.selected_index = selected_index
        else:
            self.selected_index = None

    def on_reset_intervals(self, event):
        self.intervals = fp.find_disconnects(self.recording)
        self.interval_update += 1

    def create_reset_intervals_button(self):
        reset_intervals_btn = pn.widgets.Button(name='Reset rejects',
                                                button_type='primary')
        reset_intervals_btn.on_click(self.on_reset_intervals)
        return reset_intervals_btn

    def create_tabulator_widget(self):
        # Create a Tabulator widget with the metadata_table DataFrame
        tabulator_widget = pn.widgets.Tabulator(self.metadata_table)

        # Attach the callback function to the 'selection' parameter of
        # the Tabulator widget
        tabulator_widget.param.watch(self.on_selection_change, 'selection')

        return tabulator_widget

    def update_intervals(self):
        ra = self.recording.attrs
        fp.save_rejections(self.intervals, ra['root'],
                           ra['subject'], ra['session'],
                           ra['task'], ra['run'],
                           ra['label'])
        self.interval_update += 1

    @param.depends("selected_index", "interval_update", watch=True)
    def update_regressions(self):
        rej = fp.reject(self.recording, self.intervals, fill=True)
        ch = self.recording.attrs['channel']
        config = load_preprocess_config(self.bidsroot)
        # We were doing a robust regression, but the fit isn't good enough.
        # Let's just detrend and divide by the smoothed signal instead.
        dff = fp.detrend(rej[ch], cutoff=config['detrend_cutoff'])
        dff = dff / fp.smooth(rej[ch])
        dff.name = 'dff'
        # dff = fp.series_like(self.recording, name='dff')
        # dff.loc[rej.index] = fp.detrend(rej[ch])
        # dff = dff / fp.smooth(rej[ch])
        # OLD REGRESSION CODE
        # fit = fp.fit(rej)
        # regression = fp.series_like(rej, name='regression')
        # regression[:] = fit.fittedvalues
        # self.regression = regression
        # dff = fp.series_like(self.recording, name='dff')
        # dff.loc[rej.index] = (rej[ch] - regression) / regression
        # dff = dff / dff.std()
        self.dff = dff
        self.regression_update += 1

    @param.depends("regression_update")
    def plot_all(self):
        if self.recording is None:
            return
        regression = self.regression
        tools = ['xbox_select']
        isoch = self.recording.attrs['iso_channel']
        ch = self.recording.attrs['channel']
        iso_shade = datashade(
            signal_curve(self.recording[isoch], y_dim='F'),
            aggregator=ds.count(), cmap='blue')
        sig_shade = datashade(
            signal_curve(self.recording[ch], y_dim='F'),
            aggregator=ds.count(), cmap='red').opts(tools=tools)
        dff_shade = datashade(
            signal_curve(self.dff, y_dim='dF/F'),
            aggregator=ds.count(), cmap='green')
        overlay = interval_overlay_map(iso_shade, self.intervals,
                                       self.update_intervals)
        # plot = (rej_shade.opts(xaxis=None) +
        plot = ((iso_shade * sig_shade * overlay).opts(xaxis=None) +
                dff_shade)
        plot = plot.opts(
            opts.RGB(responsive=True, min_width=600, min_height=300,
                     tools=tools))
        return plot.cols(1)

    def view(self):
        return pn.Row(
            pn.Column(
                self.create_tabulator_widget(),
                self.create_reset_intervals_button()
            ),
            pn.Column(
                self.plot_all
            ),
            styles=dict(background='WhiteSmoke')
        )
