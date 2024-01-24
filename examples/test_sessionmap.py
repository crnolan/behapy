# %%
from pathlib import Path
from tdt import read_block
from behapy.tdt import generate_session_map
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# %%
rootpath = Path('/home/cnolan/Projects/CSG010/sourcedata')

# %%
session_map = generate_session_map(
    rootpath,
    files_from='drugs/cohort-1/',
    stream_map={'_405A': ('iso', 'DMS'),
                '_465A': ('dLight', 'DMS')},
    epoc_map={'PrtA': 'DMS'})
session_map.to_csv('~/Projects/CSG010/etc/session_map.csv')

# %%
