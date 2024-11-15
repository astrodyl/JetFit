"""
    Flux.py

    Provides numerous analytical model implementations from Sari et al.
    (1998). See: https://arxiv.org/abs/astro-ph/9712005.

    All implementations ignore the self-absorption regime since it does
    not affect either the optical or the X-ray radiation in which we are
    interested.
"""


class FluxModel:
    """
    Flux Class

    Attributes
    ----------
    peak_flux : float

    cooling_frequency : float

    synchrotron_frequency : float

    electron_index : float

    """
    def __init__(self,
                 peak_flux: float,
                 cooling_frequency: float,
                 synchrotron_frequency: float,
                 electron_index: float
                 ):

        self.peak_flux = peak_flux
        self.cooling_frequency = cooling_frequency
        self.synchrotron_frequency = synchrotron_frequency
        self.electron_index = electron_index
