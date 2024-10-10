

"""
    Flux.py

    Provides numerous analytical model implementations from Sari et al.
    (1998). See: https://arxiv.org/abs/astro-ph/9712005.

    All implementations ignore the self-absorption regime since it does
    not affect either the optical or the X-ray radiation in which we are
    interested.
"""


class Flux:
    def __init__(self, **kw):
        """ Super class containing the characteristic break frequencies,
        peak flux, and spectral index.

        :param kw: keyword arguments
            :required:
                - peak_flux:
                - cooling_frequency: measured in Hz
                - synchrotron_frequency: measured in Hz
                - spectral_index:
        """
        self.peak_flux = kw.pop('peak_flux')
        self.cooling_frequency = kw.pop('cooling_frequency')
        self.synchrotron_frequency = kw.pop('synchrotron_frequency')
        self.spectral_index = kw.pop('spectral_index')


class FluxSlowCoolingSpectral(Flux):
    def __init__(self, frequency: float, **kw):
        """ Implements the slow cooling spectral flux model in section 2,
        equation 7.

        :param frequency: measured in Hz
        :param kw: keyword arguments, see super class for description
        """
        super().__init__(**kw)

        self.frequency = frequency

    def flux(self) -> float:
        """ Returns the spectral flux for a given segment as defined in
        Sari et al. (1998). Implements equation 7 in section 2.

        :return: spectral flux in units of erg / s / cm^2 / Hz
        """
        if self.frequency <= self.synchrotron_frequency:
            return self.peak_flux * (self.frequency / self.synchrotron_frequency) ** (1 / 3)

        if self.frequency < self.cooling_frequency:
            return self.peak_flux * (self.frequency / self.synchrotron_frequency) ** -((self.spectral_index - 1) / 2)

        if self.frequency >= self.cooling_frequency:
            return (self.peak_flux *
                    ((self.cooling_frequency / self.synchrotron_frequency) ** (-(self.spectral_index - 1) / 2)) *
                    ((self.frequency / self.cooling_frequency) ** (-self.spectral_index / 2)))


class FluxSlowCoolingIntegrated(Flux):
    def __init__(self, bounds: tuple, **kw):
        """ Integrates the slow cooling spectral flux model of Sari et
        al. (1998).

        :param bounds: tuple of (lower_bound, upper_bound) measured in Hz
        :param kw: keyword arguments, see super class for description
        """
        super().__init__(**kw)

        if len(bounds) != 2:
            raise ValueError('Argument `bounds` must have exactly two elements.')

        if bounds[0] > bounds[1]:
            raise ValueError('Argument `bounds` was provided in reverse order.')

        self.lower_frequency = bounds[0]
        self.upper_frequency = bounds[1]

    def integrate(self) -> float:
        """ Calculates the integrated flux for a range of frequencies by
        integrating the slow cooling set of equations (section 2, equation
        7) in Sari et al. (1998). Frequency ranges that span more than one
        segment are broken into multiple integrals.

        :return: integrated flux in units of erg / s / cm^2
        """
        if self.upper_frequency <= self.synchrotron_frequency:
            # Entire frequency range is within segment F
            return self.integrate_segment_f(self.lower_frequency, self.upper_frequency)

        if self.upper_frequency <= self.cooling_frequency:
            if self.lower_frequency < self.synchrotron_frequency:
                # Frequency range spans both segment F and G
                return (self.integrate_segment_f(self.lower_frequency, self.synchrotron_frequency) +
                        self.integrate_segment_g(self.synchrotron_frequency, self.upper_frequency))

            # Entire frequency range is within segment G
            return self.integrate_segment_g(self.lower_frequency, self.upper_frequency)

        if self.lower_frequency < self.synchrotron_frequency:
            # Frequency range spans segments F, G, and H
            return (self.integrate_segment_f(self.lower_frequency, self.synchrotron_frequency) +
                    self.integrate_segment_g(self.synchrotron_frequency, self.cooling_frequency) +
                    self.integrate_segment_h(self.cooling_frequency, self.upper_frequency))

        if self.lower_frequency < self.cooling_frequency:
            # Frequency range spans both segment G and H
            return (self.integrate_segment_g(self.lower_frequency, self.cooling_frequency) +
                    self.integrate_segment_h(self.cooling_frequency, self.upper_frequency))

        # Entire frequency range is within segment H
        return self.integrate_segment_h(self.lower_frequency, self.upper_frequency)

    def integrate_segment_f(self, lower: float, upper: float) -> float:
        """ Integrates the slow cooling flux equation for segment B in
        Sari et al. (1998). See section 2, equation 7. If you aren't
        certain if the frequency range is contained within segment B,
        use the generic `integrate` method.

        :param lower: lower integration bound
        :param upper: upper integration bound
        :return: integrated flux in units of erg / s / cm^2
        """
        return ((3 / 4) * self.peak_flux * ((1 / self.synchrotron_frequency) ** (1 / 3)) *
                (upper ** (4 / 3) - lower ** (4 / 3)))

    def integrate_segment_g(self, lower: float, upper: float) -> float:
        """ Integrates the slow cooling flux equation for segment C in
        Sari et al. (1998). See section 2, equation 7. If you aren't
        certain if the frequency range is contained within segment C,
        use the generic `integrate` method.

        :param lower: lower integration bound
        :param upper: upper integration bound
        :return: integrated flux in units of erg / s / cm^2
        """
        return ((2 / (3 - self.spectral_index)) * self.peak_flux *
                ((1 / self.synchrotron_frequency) ** (-(self.spectral_index - 1) / 2)) *
                (upper ** ((3 - self.spectral_index) / 2) - lower ** ((3 - self.spectral_index) / 2)))

    def integrate_segment_h(self, lower: float, upper: float) -> float:
        """ Integrates the slow cooling flux equation for segment D in
        Sari et al. (1998). See section 2, equation 7. If you aren't
        certain if the frequency range is contained within segment D,
        use the generic `integrate` method.

        :param lower: lower integration bound
        :param upper: upper integration bound
        :return: integrated flux in units of erg / s / cm^2
        """
        return (self.peak_flux * (2 / (2 - self.spectral_index)) *
                ((self.cooling_frequency / self.synchrotron_frequency) ** (-(self.spectral_index - 1) / 2)) *
                ((1 / self.cooling_frequency) ** (-self.spectral_index / 2)) *
                (upper ** ((2 - self.spectral_index) / 2) - lower ** ((2 - self.spectral_index) / 2)))


class FluxFastCoolingSpectral(Flux):
    def __init__(self, frequency: float, **kw):
        """ Implements the fast cooling spectral flux model in section 2,
        equation 7.

        :param frequency: measured in Hz
        :param kw: keyword arguments, see super class for description
        """
        super().__init__(**kw)

        self.frequency = frequency

    def flux(self) -> float:
        """ Returns the spectral flux for a given segment as defined in
        Sari et al. (1998). Implements equation 7 in section 2.

        :return: spectral flux in units of erg / s / cm^2 / Hz
        """
        if self.frequency <= self.cooling_frequency:
            return self.peak_flux * (self.frequency / self.cooling_frequency) ** (1 / 3)

        if self.frequency < self.synchrotron_frequency:
            return self.peak_flux * (self.frequency / self.cooling_frequency) ** (-1 / 2)

        if self.frequency >= self.synchrotron_frequency:
            return (self.peak_flux * ((self.synchrotron_frequency / self.cooling_frequency) ** (-1 / 2)) *
                    ((self.frequency / self.synchrotron_frequency) ** (-self.spectral_index / 2)))


class FluxFastCoolingIntegrated(Flux):
    def __init__(self, bounds: tuple, **kw):
        """ Integrates the fast cooling spectral flux model of Sari et
        al. (1998).

        :param bounds: tuple of (lower_bound, upper_bound) measured in Hz
        :param kw: keyword arguments, see super class for description
        """
        super().__init__(**kw)

        if len(bounds) != 2:
            raise ValueError('Argument `bounds` must have exactly two elements.')

        if bounds[0] > bounds[1]:
            raise ValueError('Argument `bounds` was provided in reverse order.')

        self.lower_frequency = bounds[0]
        self.upper_frequency = bounds[1]

    def integrate(self) -> float:
        """ Calculates the integrated flux for a range of frequencies by
        integrating the fast cooling set of equations (section 2, equation
        7) in Sari et al. (1998). Frequency ranges that span more than one
        segment are broken into multiple integrals.

        :return: integrated flux in units of erg / s / cm^2
        """
        if self.upper_frequency <= self.cooling_frequency:
            # Entire frequency range is within segment B
            return self.integrate_segment_b(self.lower_frequency, self.upper_frequency)

        if self.upper_frequency <= self.synchrotron_frequency:
            if self.lower_frequency < self.cooling_frequency:
                # Frequency range spans both segment B and C
                return (self.integrate_segment_b(self.lower_frequency, self.cooling_frequency) +
                        self.integrate_segment_c(self.cooling_frequency, self.upper_frequency))

            # Entire frequency range is within segment C
            return self.integrate_segment_c(self.lower_frequency, self.upper_frequency)

        if self.lower_frequency < self.cooling_frequency:
            # Frequency range spans segments B, C, and D
            return (self.integrate_segment_b(self.lower_frequency, self.cooling_frequency) +
                    self.integrate_segment_c(self.cooling_frequency, self.synchrotron_frequency) +
                    self.integrate_segment_d(self.synchrotron_frequency, self.upper_frequency))

        if self.lower_frequency < self.synchrotron_frequency:
            # Frequency range spans both segment C and D
            return (self.integrate_segment_c(self.lower_frequency, self.synchrotron_frequency) +
                    self.integrate_segment_d(self.synchrotron_frequency, self.upper_frequency))

        # Entire frequency range is within segment D
        return self.integrate_segment_d(self.lower_frequency, self.upper_frequency)

    def integrate_segment_b(self, lower: float, upper: float) -> float:
        """ Integrates the fast cooling flux equation for segment B in
        Sari et al. (1998). See section 2, equation 7. If you aren't
        certain if the frequency range is contained within segment B,
        use the generic `integrate` method.

        :param lower: lower integration bound
        :param upper: upper integration bound
        :return: integrated flux in units of erg / s / cm^2
        """
        return ((3 / 4) * self.peak_flux * ((1 / self.cooling_frequency) ** (1 / 3)) *
                (upper ** (4 / 3) - lower ** (4 / 3)))

    def integrate_segment_c(self, lower: float, upper: float) -> float:
        """ Integrates the fast cooling flux equation for segment C in
        Sari et al. (1998). See section 2, equation 7. If you aren't
        certain if the frequency range is contained within segment C,
        use the generic `integrate` method.

        :param lower: lower integration bound
        :param upper: upper integration bound
        :return: integrated flux in units of erg / s / cm^2
        """
        return (2 * self.peak_flux * ((1 / self.cooling_frequency) ** -(1 / 2)) *
                (upper ** (1 / 2) - lower ** (1 / 2)))

    def integrate_segment_d(self, lower: float, upper: float) -> float:
        """ Integrates the fast cooling flux equation for segment D in
        Sari et al. (1998). See section 2, equation 7. If you aren't
        certain if the frequency range is contained within segment D,
        use the generic `integrate` method.

        :param lower: lower integration bound
        :param upper: upper integration bound
        :return: integrated flux in units of erg / s / cm^2
        """
        return (self.peak_flux * (2 / (2 - self.spectral_index)) *
                ((self.synchrotron_frequency / self.cooling_frequency) ** (-1 / 2)) *
                ((1 / self.synchrotron_frequency) ** (-self.spectral_index / 2)) *
                (upper ** ((2 - self.spectral_index) / 2) - lower ** ((2 - self.spectral_index) / 2)))
