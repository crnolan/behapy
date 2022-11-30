# %%

import tdt
import os
from parameters import getParameters
from fp import *

p,sinfo = getParameters()


# %%

for f,fn in enumerate(sinfo):

    print('\n')

    fnl = os.path.join(p['dataDir'],fn)
    print('reading data from {}'.format(fnl))

    data = tdt.read_block(fnl)

    # %%
    # view the raw traces

    # %% 
    # find artefacts // discontinuities

    # %%
    # log your decisions about artefacts

    # %%
    # normalise
    dff = normalise(signal,control,mask,fs,method=p['normMethod'],detrend=p['normDetrend'])


# %%
