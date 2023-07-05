 %%
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
from tdt import read_block
import json
from collections import defaultdict
from behapy.tdt import get_epoch_df, load_event_names, convert_block
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
PROJECTPATH = Path('..')
SOURCEPATH = PROJECTPATH / 'sourcedata/'
RAWPATH = PROJECTPATH / 'rawdata/'


# %%
session_info = pd.read_csv(SOURCEPATH / 'session-map.csv')

# %%
block = read_block(SOURCEPATH / session_info.iloc[0].block)

# %%
get_epoch_df(block.epocs['PrtA'])

# %%
convert_block(session_info, RAWPATH)

