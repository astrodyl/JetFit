from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux_value import SpectralFluxValue
from jetfit.model.spectral.sf_model import SpectralFluxModel


class SFCFModel(SpectralFluxModel):
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
        if self.frequency <= self.cooling_frequency:
            return self.regime_b_flux(desired_u)

        if self.frequency < self.synchrotron_frequency:
            return self.regime_c_flux(desired_u)

        if self.frequency >= self.synchrotron_frequency:
            return self.regime_d_flux(desired_u)

    def regime_b_flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Calculates and returns a flux in the slow cooling regime (B).

        Parameters
        ----------
        desired_u : FluxUnits
            The units of the returned flux.

        Returns
        -------
        SpectralFluxValue
            The modeled flux in the slow cooling regime (B).
        """
        flux = self.peak_flux * (self.frequency / self.cooling_frequency) ** (1 / 3)

        return self.to_flux_object(flux, desired_u)

    def regime_c_flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
        """
        Calculates and returns a flux in the slow cooling regime (C).

        Parameters
        ----------
        desired_u : FluxUnits
            The units of the returned flux.

        Returns
        -------
        SpectralFluxValue
            The modeled flux in the slow cooling regime (C).
        """
        flux = self.peak_flux * (self.frequency / self.cooling_frequency) ** (-1 / 2)

        return self.to_flux_object(flux, desired_u)

    def regime_d_flux(self, desired_u: FluxUnits) -> SpectralFluxValue:
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
        flux = (self.peak_flux * ((self.synchrotron_frequency / self.cooling_frequency) ** (-1 / 2)) *
                ((self.frequency / self.synchrotron_frequency) ** (-self.electron_index / 2)))

        return self.to_flux_object(flux, desired_u)