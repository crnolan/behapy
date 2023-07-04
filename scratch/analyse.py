# %%
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import behapy.fp as fp
from behapy.events import load_events, find_events
from behapy.pathutils import list_preprocessed
import statsmodels.api as sm
import holoviews as hv
from holoviews import opts
import datashader as ds
from holoviews.operation.datashader import datashade, dynspread, rasterize
import panel as pn

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
BIDSROOT = Path('/scratch/cnolan/TB006')
recordings = list_preprocessed(BIDSROOT)
