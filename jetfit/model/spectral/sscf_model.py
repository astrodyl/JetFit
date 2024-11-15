from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux_value import SpectralFluxValue
from jetfit.model.spectral.sf_model import SpectralFluxModel


class SSCFModel(SpectralFluxModel):
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
            electron_index,
            frequency
        )

    def flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Calculates and returns a flux in the determined slow cooling regime.

        Parameters
        ----------
        desired_u : FluxUnits
            The units of the returned flux.

        Returns
        -------
        SpectralFluxValue
            The modeled flux in the slow cooling regime.
        """
        if self.frequency <= self.synchrotron_frequency:
            return self.regime_f_flux(desired_u)

        if self.frequency < self.cooling_frequency:
            return self.regime_g_flux(desired_u)

        if self.frequency >= self.cooling_frequency:
            return (self.peak_flux *
                    ((self.cooling_frequency / self.synchrotron_frequency) ** (-(self.electron_index - 1) / 2)) *
                    ((self.frequency / self.cooling_frequency) ** (-self.electron_index / 2)))

    def regime_f_flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Calculates and returns a flux in the slow cooling regime (F).

        Parameters
        ----------
        desired_u : FluxUnits
            The units of the returned flux.

        Returns
        -------
        SpectralFluxValue
            The modeled flux in the slow cooling regime (F).
        """
        flux = self.peak_flux * (self.frequency / self.synchrotron_frequency) ** (1 / 3)

        return self.to_flux_object(flux, desired_u)

    def regime_g_flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Calculates and returns a flux in the slow cooling regime (G).

        Parameters
        ----------
        desired_u : FluxUnits
            The units of the returned flux.

        Returns
        -------
        SpectralFluxValue
            The modeled flux in the slow cooling regime (G).
        """
        flux = self.peak_flux * (self.frequency / self.synchrotron_frequency) ** -((self.electron_index - 1) / 2)

        return self.to_flux_object(flux, desired_u)

    def regime_h_flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Calculates and returns a flux in the slow cooling regime (D).

        Parameters
        ----------
        desired_u : FluxUnits
            The units of the returned flux.

        Returns
        -------
        SpectralFluxValue
            The modeled flux in the slow cooling regime (D).
        """
        eq1 = self.peak_flux
        eq2 = (self.cooling_frequency / self.synchrotron_frequency) ** (-(self.electron_index - 1) / 2)
        eq3 = (self.frequency / self.cooling_frequency) ** (-self.electron_index / 2)

        return self.to_flux_object(eq1*eq2*eq3, desired_u)
