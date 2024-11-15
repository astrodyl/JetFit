from jetfit.core.defns.enums import FluxUnits
from jetfit.core.utilities import maths
from jetfit.core.values.flux_value import IntegratedFluxValue
from jetfit.model.flux_model import FluxModel


class IntegratedFluxModel(FluxModel):
    """
    ???

    Attributes
    ----------
    lower_frequency : float
        The lower_bound measured in Hz.

    upper_frequency : float
        The upper_bound measured in Hz.
    """
    def __init__(self,
                 peak_flux: float,
                 cooling_frequency: float,
                 synchrotron_frequency: float,
                 electron_index: float,
                 lower_frequency: float,
                 upper_frequency: float):

        super().__init__(
            peak_flux,
            cooling_frequency,
            synchrotron_frequency,
            electron_index
        )
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

    def integrate(self, _: FluxUnits) -> IntegratedFluxValue:
        """
        Subclasses require an integrate method.

        Parameters
        ----------
        _ : FluxUnits
            The units of the returned flux.

        Returns
        -------
        IntegratedFluxValue
            The modeled flux in the provided units.
        """
        raise NotImplementedError('Missing integrate method.')

    def to_flux_object(self, value: float, desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Converts a flux float to an `IntegratedFluxValue`.

        Parameters
        ----------
        value : float

        desired_u : `jetfit.core.defns.enums.FluxUnits`

        Returns
        -------
        IntegratedFluxValue
        """
        flux = maths.convert_flux_to(
            value=value,
            from_u=FluxUnits.MJY,
            to_u=desired_u
        )

        return IntegratedFluxValue(
            value=flux,
            lower=0.0,
            upper=0.0,
            units=desired_u,
            frequency_range=(
                self.lower_frequency,
                self.upper_frequency
            )
        )
