from jetfit.core.defns.enums import FluxUnits
from jetfit.core.values.flux_value import IntegratedFluxValue
from jetfit.model.integrated.if_model import IntegratedFluxModel


class IFCFModel(IntegratedFluxModel):
    """
    Integrated Fast Cooling Flux Model
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
        Models a fast cooling flux for the Sari & Piran model.

        Parameters
        ----------
        desired_u : `jetfit.core.defns.enums.FluxUnits`
            The units of the returned flux.

        References
        ----------
        ..[1]

        Returns
        -------
        IntegratedFluxValue
            The integrated flux value in the fast cooling regime (B).
        """
        if self.upper_frequency <= self.cooling_frequency:
            # Entire frequency range is within segment B
            return self.integrate_segment_b(self.lower_frequency, self.upper_frequency, desired_u)

        if self.upper_frequency <= self.synchrotron_frequency:
            if self.lower_frequency < self.cooling_frequency:
                # Frequency range spans both segment B and C
                return (self.integrate_segment_b(self.lower_frequency,self.cooling_frequency,desired_u) +
                        self.integrate_segment_c(self.cooling_frequency, self.upper_frequency, desired_u))

            # Entire frequency range is within segment C
            return self.integrate_segment_c(self.lower_frequency, self.upper_frequency, desired_u)

        if self.lower_frequency < self.cooling_frequency:
            # Frequency range spans segments B, C, and D
            return (self.integrate_segment_b(self.lower_frequency, self.cooling_frequency, desired_u) +
                    self.integrate_segment_c(self.cooling_frequency, self.synchrotron_frequency, desired_u) +
                    self.integrate_segment_d(self.synchrotron_frequency, self.upper_frequency, desired_u))

        if self.lower_frequency < self.synchrotron_frequency:
            # Frequency range spans both segment C and D
            return (self.integrate_segment_c(self.lower_frequency, self.synchrotron_frequency, desired_u) +
                    self.integrate_segment_d(self.synchrotron_frequency, self.upper_frequency, desired_u))

        # Entire frequency range is within segment D
        return self.integrate_segment_d(self.lower_frequency, self.upper_frequency, desired_u)

    def integrate_segment_b(self, lower: float, upper: float,
                            desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Models a fast cooling flux for the Sari & Piran segment (B) _[1].

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
            The integrated flux value in the fast cooling regime (B).
        """
        eq1 = (3 / 4) * self.peak_flux
        eq2 = (1 / self.cooling_frequency) ** (1 / 3)
        eq3 = (upper ** (4 / 3)) - (lower ** (4 / 3))

        return self.to_flux_object(eq1*eq2*eq3, desired_u)

    def integrate_segment_c(self, lower: float, upper: float,
                            desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Models a fast cooling flux for the Sari & Piran segment (C) _[1].

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
            The integrated flux value in the fast cooling regime (C).
        """
        eq1 = 2 * self.peak_flux
        eq2 = (1 / self.cooling_frequency) ** -0.5
        eq3 = (upper ** 0.5) - (lower ** 0.5)

        return self.to_flux_object(eq1*eq2*eq3, desired_u)

    def integrate_segment_d(self, lower: float, upper: float,
                            desired_u: FluxUnits) -> IntegratedFluxValue:
        """
        Models a fast cooling flux for the Sari & Piran segment (D) _[1].

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
            The integrated flux value in the fast cooling regime (D).
        """
        eq1 = self.peak_flux * (2 / (2 - self.electron_index))
        eq2 = (self.synchrotron_frequency / self.cooling_frequency) ** -0.5
        eq3 = (1 / self.synchrotron_frequency) ** (-self.electron_index / 2)
        eq4 = (upper ** ((2 - self.electron_index) / 2)) - (lower ** ((2 - self.electron_index) / 2))

        return self.to_flux_object(eq1*eq2*eq3*eq4, desired_u)
