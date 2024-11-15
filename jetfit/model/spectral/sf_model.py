from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux_value import SpectralFluxValue
from jetfit.core.utilities import maths
from jetfit.model.flux_model import FluxModel


class SpectralFluxModel(FluxModel):
    """

    """
    def __init__(self,
                 peak_flux: float,
                 cooling_frequency: float,
                 synchrotron_frequency: float,
                 electron_index: float,
                 frequency: float):

        super().__init__(
            peak_flux,
            cooling_frequency,
            synchrotron_frequency,
            electron_index
        )
        self.frequency = frequency

    def flux(self, units: FluxUnits):
        """ Subclasses require a flux method. """
        raise NotImplementedError('Missing flux method.')

    def to_flux_object(self, value: float, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Converts a flux float to an `SpectralFluxValue`.

        Parameters
        ----------
        value : float

        desired_u : `jetfit.core.defns.enums.FluxUnits`

        Returns
        -------
        SpectralFluxValue
        """
        flux = maths.convert_flux_to(
            value=value,
            from_u=FluxUnits.MJY,
            to_u=desired_u
        )

        return SpectralFluxValue(
            value=flux,
            lower=0.0,
            upper=0.0,
            units=desired_u,
            frequency=self.frequency
        )