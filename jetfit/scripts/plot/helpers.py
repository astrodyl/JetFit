import numpy as np


OPTION_MAP = {
    # JC Optical/NIR (circles)
    'U': {'color': '#8601AF', 'marker': '.'},
    'B': {'color': '#0247FE', 'marker': '.'},
    'V': {'color': '#66B032', 'marker': '.'},
    'R': {'color': '#FE2712', 'marker': '.'},
    'I': {'color': '#4424D6', 'marker': '.'},
    'J': {'color': '#66B032', 'marker': '.'},
    'H': {'color': '#FC600A', 'marker': '.'},
    'K': {'color': '#FE2712', 'marker': '.'},

    # SDSS Optical (squares)
    'u': {'color': 'tab:purple', 'marker': 's'},
    'g': {'color': 'tab:blue',   'marker': 's'},
    'r': {'color': 'tab:orange', 'marker': 's'},
    'i': {'color': 'tab:red',    'marker': 's'},
    'z': {'color': 'tab:pink',   'marker': 's'},

    # Swift Optical/UV/XRAY (diamonds, hexagons)
    'uvot-u': {'color': 'cyan',       'marker': '.'},
    'uvot-b': {'color': 'lightblue',  'marker': '.'},
    'uvot-v': {'color': 'lightgreen', 'marker': '.'},
    'uvw2': {'color': 'pink',         'marker': '.'},
    'uvm2': {'color': 'darkblue',     'marker': '.'},
    'uvw1': {'color': 'green',        'marker': '.'},
    'xray': {'color': 'black',        'marker': '.'},

    # HST
    'F775W': {'color': 'yellow', 'marker': '.'},
    'F125W': {'color': 'grey',   'marker': '.'},

    # Radio
    'C': {'color': 'royalblue', 'marker': '.'},
    'C2': {'color': 'purple', 'marker': '.'},
    'Ka': {'color': 'peachpuff', 'marker': '.'},
    'Kb': {'color': 'peru', 'marker': '.'},
    'Kc': {'color': 'palevioletred', 'marker': '.'},
    'Kd': {'color': 'lightcoral', 'marker': '.'},
    'W': {'color': 'teal', 'marker': '.'},
    'S': {'color': 'teal', 'marker': '.'},
}

# Aliases
OPTION_MAP['Rc'] = OPTION_MAP['R']
OPTION_MAP['Ic'] = OPTION_MAP['I']
OPTION_MAP['Ks'] = OPTION_MAP['K']
OPTION_MAP['uprime'] = OPTION_MAP['u']
OPTION_MAP['gprime'] = OPTION_MAP['g']
OPTION_MAP['rprime'] = OPTION_MAP['r']
OPTION_MAP['iprime'] = OPTION_MAP['i']
OPTION_MAP['zprime'] = OPTION_MAP['z']


def sec_to_days(x):
    """ Used for plotting axes. """
    return x / 86400


def days_to_sec(x):
    """ Used for plotting axes. """
    return x * 86400
