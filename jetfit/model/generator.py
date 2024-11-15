from jetfit.core.defns.enums import FluxType
from jetfit.model.integrated.ifcf_model import IFCFModel
from jetfit.model.integrated.iscf_model import ISCFModel
from jetfit.model.spectral.sfcf_model import SFCFModel
from jetfit.model.spectral.sscf_model import SSCFModel


def generate(observed, peak_flux: float, cooling_frequency: float,
             synchrotron_frequency: float, electron_index: float):
    """

    Returns
    -------

    Raises
    ------
    TypeError
        If an unsupported flux type is encountered.
    """
    if observed.type == FluxType.SPECTRAL:
        return spectral_flux_factory(
            peak_flux,
            cooling_frequency,
            synchrotron_frequency,
            electron_index,
            observed.frequency
        ).flux(observed.units)
    elif observed.type == FluxType.INTEGRATED:
        return integrated_flux_factory(
            peak_flux,
            cooling_frequency,
            synchrotron_frequency,
            electron_index,
            observed.frequency_range
        ).integrate(observed.units)
    else:
        raise TypeError(f'Unsupported flux type: {observed.type}')


def integrated_flux_factory(peak_flux: float,
                            cooling_frequency: float,
                            synchrotron_frequency: float,
                            electron_index: float,
                            bounds: tuple[float, float]):
    """
    Returns the

    Parameters
    ----------
    peak_flux
    cooling_frequency
    synchrotron_frequency
    electron_index
    bounds

    Returns
    -------

    """
    if synchrotron_frequency > cooling_frequency:
        model = IFCFModel
    else:
        model = ISCFModel

    return model(
        peak_flux,
        cooling_frequency,
        synchrotron_frequency,
        electron_index,
        bounds[0],
        bounds[1]
    )


def spectral_flux_factory(peak_flux: float,
                          cooling_frequency: float,
                          synchrotron_frequency: float,
                          electron_index: float,
                          frequency: float):
    """

    Parameters
    ----------
    peak_flux
    cooling_frequency
    synchrotron_frequency
    electron_index
    frequency

    Returns
    -------

    """
    if synchrotron_frequency > cooling_frequency:
        model = SFCFModel
    else:
        model = SSCFModel

    return model(
        peak_flux,
        cooling_frequency,
        synchrotron_frequency,
        electron_index,
        frequency
    )
