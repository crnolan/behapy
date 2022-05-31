from behapy.tdt import Event

EVENTS = {
    'MagEntry': Event('MagEntry', ['Mtry', 'MMag'], [8], True, False),
    'Pellet': Event('Pellet', ['Plet', 'Ppel'], [18], True, False),
    'LeverPress': Event('LeverPress', ['Less', 'LLev', 'LLPr'], [18], True, False),
    'LeverIn': Event('LeverIn', ['LLIn'], [18], True, False),
    'LeverOut': Event('LeverOut', ['LLOut'], [18], True, False)
}
STREAMS = {
    'dLight': '_465P',
    'dLight-iso': '_405P'
}
DLIGHT = '_465P'
ISO = '_405P'
DFFRT = 'dFFa'
