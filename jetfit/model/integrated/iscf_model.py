from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux_value import IntegratedFluxValue
from jetfit.model.integrated.if_model import IntegratedFluxModel


class ISCFModel(IntegratedFluxModel):
    """
    Integrate Slow Cooling Flux Model
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
            electron_index,
            lower_frequency,
            upper_frequency
        )

    def integrate(self, desired_u: FluxUnits):
        """
        Models a slow cooling flux for the Sari & Piran model.

        Parameters
        ----------
        desired_u : `jetfit.core.defns.enums.FluxUnits`


        Returns
        -------
        IntegratedFluxValue
            The modeled integrated flux value.
        """
        if self.upper_frequency <= self.synchrotron_frequency:
            # Entire frequency range is within segment F
            return self.integrate_segment_f(self.lower_frequency, self.upper_frequency, desired_u)

        if self.upper_frequency <= self.cooling_frequency:
            if self.lower_frequency < self.synchrotron_frequency:
                # Frequency range spans both segment F and G
                return (self.integrate_segment_f(self.lower_frequency, self.synchrotron_frequency, desired_u) +
                        self.integrate_segment_g(self.synchrotron_frequency, self.upper_frequency, desired_u))

            # Entire frequency range is within segment G
            return self.integrate_segment_g(self.lower_frequency, self.upper_frequency, desired_u)

        if self.lower_frequency < self.synchrotron_frequency:
            # Frequency range spans segments F, G, and H
            return (self.integrate_segment_f(self.lower_frequency, self.synchrotron_frequency, desired_u) +
                    self.integrate_segment_g(self.synchrotron_frequency, self.cooling_frequency, desired_u) +
                    self.integrate_segment_h(self.cooling_frequency, self.upper_frequency, desired_u))

        if self.lower_frequency < self.cooling_frequency:
            # Frequency range spans both segment G and H
            return (self.integrate_segment_g(self.lower_frequency, self.cooling_frequency, desired_u) +
                    self.integrate_segment_h(self.cooling_frequency, self.upper_frequency, desired_u))

        # Entire frequency range is within segment H
        return self.integrate_segment_h(self.lower_frequency, self.upper_frequency, desired_u)

    def integrate_segment_f(self, lower: float, upper: float,
                            desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Models a fast cooling flux for the Sari & Piran segment (F) _[1].

        Parameters
        ----------
        lower : float
            The lower frequency in Hz.

        upper : float
            The upper frequency in Hz.

        desired_u : `jetfit.core.defns.enums.FluxUnits`
            The units of the returned flux.

        References
        ----------
        ..[1]

        Returns
        -------
        IntegratedFluxValue
            The integrated flux value in the fast cooling regime (F).
        """
        eq1 = (3 / 4) * self.peak_flux
        eq2 = (1 / self.synchrotron_frequency) ** (1 / 3)
        eq3 = (upper ** (4 / 3)) - (lower ** (4 / 3))

        return self.to_flux_object(eq1*eq2*eq3, desired_u)

    def integrate_segment_g(self, lower: float, upper: float,
                            desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Models a fast cooling flux for the Sari & Piran segment (G) _[1].

        Parameters
        ----------
        lower : float
            The lower frequency in Hz.

        upper : float
            The upper frequency in Hz.

        desired_u : `jetfit.core.defns.enums.FluxUnits`
            The units of the returned flux.

        References
        ----------
        ..[1]

        Returns
        -------
        IntegratedFluxValue
            The integrated flux value in the fast cooling regime (G).
        """
        eq1 = (2 / (3 - self.electron_index)) * self.peak_flux
        eq2 = (1 / self.synchrotron_frequency) ** (-(self.electron_index - 1) / 2)
        eq3 = (upper ** ((3 - self.electron_index) / 2)) - (lower ** ((3 - self.electron_index) / 2))

        return self.to_flux_object(eq1*eq2*eq3, desired_u)

    def integrate_segment_h(self, lower: float, upper: float,
                            desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Models a fast cooling flux for the Sari & Piran segment (H) _[1].

        Parameters
        ----------
        lower : float
            The lower frequency in Hz.

        upper : float
            The upper frequency in Hz.

        desired_u : `jetfit.core.defns.enums.FluxUnits`
            The units of the returned flux.

        References
        ----------
        ..[1]

        Returns
        -------
        IntegratedFluxValue
            The integrated flux value in the fast cooling regime (H).
        """
        eq1 = self.peak_flux * (2 / (2 - self.electron_index))
        eq2 = (self.cooling_frequency / self.synchrotron_frequency) ** (-(self.electron_index - 1) / 2)
        eq3 = (1 / self.cooling_frequency) ** (-self.electron_index / 2)
        eq4 = (upper ** ((2 - self.electron_index) / 2)) - (lower ** ((2 - self.electron_index) / 2))

        return self.to_flux_object(eq1*eq2*eq3*eq4, desired_u)
