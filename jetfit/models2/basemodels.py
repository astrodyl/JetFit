import math

import astropy.units as u
import astropy.constants as const
import numpy as np

from jetfit.core.values import SpectralFlux, IntegratedFlux, SpectralIndex
from jetfit.mcmc.parameters.parameters import Parameters


class BaseFluxModel:
    """

    Attributes
    ----------
    f_peak : float or np.ndarray or u.Quantity['spectral flux density']
        The peak flux. If a simple float is provided,
        assumes it is measured in mJy.

    nu_m : float or np.ndarray or u.Quantity['frequency']
        The synchrotron frequency. If a simple float
        is provided, assumes it is measured in Hz.

    nu_c : float or np.ndarray or u.Quantity['frequency']
        The cooling frequency. If a simple float
        is provided, assumes it is measured in Hz.

    p : float
        The electron energy power-law index.

    k : float
        The circumburst density power-law index.
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        if isinstance(nu_m, np.ndarray) or isinstance(nu_c, np.ndarray):
            if nu_m.size != nu_c.size:
                raise ValueError('nu_m, nu_c must have the same size.')

        self.f_peak = f_peak
        self.nu_m = nu_m
        self.nu_c = nu_c
        self.p = p
        self.k = k

    @property
    def f_peak(self) -> float | np.ndarray:
        """ Returns the peak flux in mJy. """
        return self._f_peak

    @f_peak.setter
    def f_peak(self, val) -> None:
        """
        Sets the peak flux in mJy.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['spectral flux density']
            The peak flux. If a simple float is provided,
            assumes it is measured in mJy.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('mJy')
        self._f_peak = val

    @property
    def nu_m(self) -> float | np.ndarray:
        """ Returns the synchrotron frequency in Hz. """
        return self._nu_m

    @nu_m.setter
    def nu_m(self, val) -> None:
        """
        Sets synchrotron frequency in Hz.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['frequency']
            The synchrotron frequency. If a simple float
            is provided, assumes it is measured in Hz.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('Hz')
        self._nu_m = val

    @property
    def nu_c(self) -> float | np.ndarray:
        """ Returns the synchrotron frequency in Hz. """
        return self._nu_c

    @nu_c.setter
    def nu_c(self, val) -> None:
        """
        Sets cooling frequency in Hz.

        Parameters
        ----------
        val : float or np.ndarray or u.Quantity['frequency']
            The cooling frequency. If a simple float is
            provided, assumes it is measured in Hz.
        """
        if isinstance(val, u.Quantity):
            val = val.to_value('Hz')
        self._nu_c = val


class SpectralFluxModel(BaseFluxModel):
    """
    Spectral Fireball Flux Model
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        super().__init__(nu_m, nu_c, f_peak, p, k)

    def __call__(self, nu):
        """ Calls the `evaluate` method. """
        return self.evaluate(nu)

    def model(self, val: SpectralFlux) -> float:
        """
        Models a `SpectralFlux` value using its frequency.

        Parameters
        ----------
        val : SpectralFlux
            The spectral flux value to model.

        Returns
        -------
        float
            The modeled spectral flux value.
        """
        return self.evaluate(val.frequency.value)

    def evaluate(self, nu):
        """
        Calculates the smoothed flux at a given frequency, `nu`.

        Supports four cases:
            (1) One `nu` and many spectral functions:
                Returns an array of flux with length of the
                spectral functions (i.e., nu_m.size).

            (2) Many `nu` and one spectral function:
                Returns an array of flux with length of `nu`.

            (3) Many `nu` and many spectral functions:
                All arrays must be of the same size and the
                returned array will have the same size.

            (4) One `nu` and one spectral function:
                Returns a single flux value.

        Smoothing factors are taken from Granot & Sari 2002.
        Spectral indices are taken from Sari, Piran, & Narayan 1998.

        Parameters
        ----------
        nu : float or np.ndarray
            The frequency to evaluate.

        Returns
        -------
        float or np.ndarray of float
            The modeled flux with units of [mJy].

        References
        ----------
        [1] Granot & Sari 2002
        https://iopscience.iop.org/article/10.1086/338966

        [2] Sari, Piran, & Narayan 1998
        https://iopscience.iop.org/article/10.1086/311269/pdf
        """
        nu = np.atleast_1d(nu)
        nu_m = np.atleast_1d(self.nu_m)
        nu_c = np.atleast_1d(self.nu_c)

        # Handles the cases for varying sizes of inputs
        size = nu_m.size if nu.size == 1 else nu.size

        if nu.size > 1 and nu_c.size == 1:
            nu_c = np.full(nu.size, nu_c[0])
            nu_m = np.full(nu.size, nu_m[0])

        # Initialize spectral indices with slow-cooling params
        b1 = np.full(size, 1 / 3)
        b2 = np.full(size, (1 - self.p) / 2)
        b3 = np.full(size, -self.p / 2)

        # Initialize smoothing factors with slow-cooling params
        s12 = np.full(size, 1.84 - (0.040 * self.k) - (0.40 - 0.010 * self.k) * self.p)
        s23 = np.full(size, 1.15 - (0.125 * self.k) - (0.06 - 0.015 * self.k) * self.p)

        # Initialize critical frequencies in slow-cooling order
        nu12 = np.array(nu_m, copy=True)
        nu23 = np.array(nu_c, copy=True)

        # Overwrite with any fast-cooling parameters
        fast_regime = nu_m > nu_c

        if fast_regime.any():
            b2[fast_regime] = -0.5
            s12[fast_regime] = 0.597
            nu12[fast_regime] = nu_c[fast_regime]
            nu23[fast_regime] = nu_m[fast_regime]
            s23[fast_regime] = 3.34 + 0.17 * self.k - (0.82 + 0.035 * self.k) * self.p

        # return spectral flux smoothed across segments [mJy]
        return self.f_peak * (
            (((nu / nu12) ** -(s12 * (b1 - b2)) + 1) ** (s23 / s12)) *
            ((nu / nu12) ** -(s23 * b2)) +
            (((nu23 / nu12) ** -(s23 * b2)) * ((nu/nu23) ** -(s23 * b3)))
        ) ** -(1 / s23)


class IntegratedFluxModel(BaseFluxModel):
    """
    Integrated Fireball Flux Model

    Provides methods for calculating integrated fluxes
    using the spectrum for the GRB fireball model.
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        super().__init__(nu_m, nu_c, f_peak, p, k)

    def __call__(self, lower, upper):
        """ Calls the `evaluate` method. """
        return self.evaluate(lower, upper)

    def model(self, val: IntegratedFlux):
        """
        Models an `IntegratedFlux` value using its integration
        range.

        Parameters
        ----------
        val : IntegratedFlux
            The Integrated flux value to model.

        Returns
        -------
        float
            The modeled integrated flux value.
        """
        return self.evaluate(
            lower=val.int_range.lower.value,
            upper=val.int_range.upper.value
        )

    def evaluate(self, lower: float, upper: float):
        """
        Evaluates the integrated flux model using the
        `lower` and `upper` integration limits.

        Parameters
        ----------
        lower : float or np.ndarray of float
            The lower integration limit measured in Hz.

        upper : float or np.ndarray of float
            The upper integration limit measured in Hz.

        Returns
        -------
        float or np.ndarray of float
            The integrated flux with units of erg cm-2 s-1.
        """
        beta = SpectralIndexModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k
        ).evaluate(lower, upper)

        flux = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k
        ).evaluate(lower)

        # return smoothed integrated flux [erg cm-2 s-1]
        return 1e-26 * (
            (flux * lower / (beta + 1)) *
            (((upper / lower) ** (beta + 1)) - 1)
        )


class SpectralIndexModel(BaseFluxModel):
    """
    Spectral Index Model
    """
    def __init__(self, nu_m, nu_c, f_peak, p, k):
        super().__init__(nu_m, nu_c, f_peak, p, k)

    def __call__(self, lower, upper):
        """ Calls the `evaluate` method. """
        return self.evaluate(lower, upper)

    def model(self, val: SpectralIndex):
        """
        Models a `SpectralIndex` value using its integration
        limits.

        Parameters
        ----------
        val : SpectralIndex
            The spectral index value to model.

        Returns
        -------
        float
            The modeled spectral index value.
        """
        return self.evaluate(
            lower=val.int_range.lower.value,
            upper=val.int_range.upper.value,
        )

    def evaluate(self, lower, upper):
        """
        Approximates the spectral index using a two
        point approximation.

        Parameters
        ----------
        lower : float or np.ndarray of float
            The lower integration limit.

        upper : float or np.ndarray of float
            The upper integration limit.

        Returns
        -------
        float or np.ndarray of float
            The modeled spectral index.
        """
        model = SpectralFluxModel(
            self.nu_m, self.nu_c, self.f_peak, self.p, self.k)

        # return spectral index [dimension less]
        return (
            np.log10(model(upper) / model(lower)) /
            np.log10(upper / lower)
        )


class BaseSpectralModel:
    """
    Base Spectral Model. Not intended for direct use.

    Parameters
    ----------
    E : float or u.Quantity['energy']
        The explosion energy. If a float is provided, assumes
        that the value is already normalized to 1e52 erg. If
        passing a `Quantity`, it will be normalized before
        storing it as a float.

    eps_b : float
        The fraction of thermal energy in the magnetic field.

    k : float
        The density power-law index.

    z : float
        The redshift.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, eps_b, k, z):
        self.E = E
        self.eps_b = eps_b
        self.k = k
        self.z = z

    def __repr__(self) -> str:
        """ Returns readable string for printing. """
        name = self.__class__.__name__
        return f"{name}(E={self.E}, z={self.z}, k={self.k})"

    def __call__(self, *args, **kwargs):
        """ Wrapper for the evaluate method. """
        return self.evaluate(*args, **kwargs)

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the explosion energy normalized to 10e52 erg. """
        return self._E

    # noinspection PyPep8Naming
    @E.setter
    def E(self, e: float | u.Quantity) -> None:
        """
        Sets the explosion energy normalized to 10e52 erg.

        Parameters
        ----------
        e : float or astropy.units.Quantity
            The explosion energy. If a float is provided, assumes
            that the value is already normalized to 1e52 erg.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e52

        self._E = e

    @property
    def alpha(self) -> float:
        """ Returns the temporal coefficient. """
        return 16 / (17 - 4 * self.k)

    @property
    def beta(self) -> float:
        """ Returns the spectral coefficient. """
        return 4 - self.k

    def evaluate(self, *args, **kwargs):
        """ Placeholder evaluate method. """
        raise NotImplementedError(f'evaluate not implemented.')


class PeakFluxModel(BaseSpectralModel):
    """
    Peak flux model. Assumes an ultra-relativistic shock moving
    through an external medium with rho = rho0 * R^-k density.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, dL, z, k, X):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0
        self.dL = dL
        self.X = X

    @property
    def n_p(self):
        """ Returns the inverse particle density. """
        return 0.5 * (1 + self.X)

    def evaluate(self, t):
        """
        Calculates the peak flux at time `t` for a shock's
        movement that is described by `evo`.

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.array of u.Quantity['time']
            The peak flux at time `t` measured in mJy.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # Convenience variables
        k, x = self.k, 4 - self.k

        # Evaluate exponents once
        exp_z   = 0.5 * (8 - k) / x
        exp_c   = -0.5 * (24 - 7 * k) / x
        exp_en  = 0.5 * (8 - 3 * k) / x
        exp_t   = -0.5 * k / x
        exp_rho = 2 / x

        # Exponents in log-space to prevent overflow
        log_pot = (
            (10 * exp_c) +              # speed of light [cm]
            (52 * exp_en) +             # 1e52 erg normalization
            ((17 * k - 24) * exp_rho) + # proton mass [g] and radius normalization
            (4 * exp_t) -               # time conversion (d -> s)
            8.0                         # e(q_e)^3 * e(m_e)^-1 * e(m_p)^-1 - e(dL)^2 + e(cgs->mJy)
                                        # = -30 + 28 + 24 -56 + 26 = -8
        )

        # return peak flux [mJy]
        return (
            # k-independent mantissas
            13.71383 *  # = 4/3 * sqrt(2) * m(q_e)^3 * m(m_e)^-1 * m(m_p)^-1

            # k-dependent mantissas
            (2.9979 ** exp_c) *     # speed of light [cm]
            (1.67262 ** exp_rho) *  # density normalization [g]
            (8.64 ** exp_t) *       # time conversion (d -> s)

            # k-dependent terms
            (math.pi ** -((2 - k) / x)) *
            (self.alpha ** -(0.5 * (8 - 3 * k) / x)) *
            (self.beta ** -(0.5 * k / x)) *

            # model parameters
            (self.eps_b ** 0.5) *       # magnetic field fraction
            ((1 + self.z) ** exp_z) *   # redshift
            (self.E ** exp_en) *        # explosion energy / 1e52 erg
            self.n_p *                  # particle density
            (self.rho0 ** exp_rho) *    # number density / R_*
            (self.dL ** -2) *           # luminosity distance / 1e28 cm
            (t ** exp_t) *              # time in days

            # exponents in linear-space
            (10 ** log_pot)
        )


class CoolingFrequencyModel(BaseSpectralModel):
    """
    Cooling frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_b, k, z):
        super().__init__(E, eps_b, k, z)
        self.rho0 = rho0

    def evaluate(self, t):
        """
        Calculates the cooling frequency at time `t`
        for a shock's movement that is described by `evo`.

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.array of float
            The cooling frequency at time `t` measured in Hz.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # convenience variables
        k, x = self.k, 4 - self.k

        # evaluate exponents once
        exp_c   = 0.5 * (68 - 19 * k) / x
        exp_en  = -0.5 * (4 - 3 * k) / x
        exp_t   = -0.5 * (4 - 3 * k) / x
        exp_z   = -0.5 * (4 + k) / x
        exp_rho = -4 / x

        # exponents in log-space to prevent overflow
        log_pot = (
            (10 * exp_c) + (52 * exp_en) - 70 +
            (4 * exp_t) + ((17 * k - 24) * exp_rho)
        )

        # return cooling frequency [Hz]
        return (
            # k-independent mantissas
            0.014871 *  # 81/8192 * sqrt(2) * m(q_e)^-7 * m(m_e)^5

            # k-dependent mantissas
            (2.9979 ** exp_c) *     # speed of light
            (1.67262 ** exp_rho) *  # density normalization
            (8.64 ** exp_t) *       # time conversion (d -> s)

            # k-dependent terms
            (math.pi ** -((8 - k) / x)) *
            (self.alpha ** (0.5 * (4 - 3 * k) / x)) *
            (self.beta ** (0.5 * (12 - k) / x)) *

            # model parameters
            ((1 + self.z) ** exp_z) *   # redshift
            (self.eps_b ** -1.5) *      # magnetic field fraction
            (self.E ** exp_en) *        # explosion energy
            (self.rho0 ** exp_rho) *    # density normalization
            (t ** exp_t) *              # time in days

            # exponents in linear-space
            (10 ** log_pot)
        )


class SynchrotronFrequencyModel(BaseSpectralModel):
    """
    Synchrotron frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    Attributes
    ----------
    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons, unit=None.

    X : float
        The hydrogen mass fraction, unit=None.

    p : float
        The electron energy power-law index, unit=None.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, eps_e, eps_b, k, z, X, p):
        super().__init__(E, eps_b, k, z)
        self.eps_e = eps_e
        self.X = X
        self.p = p

    @property
    def n_p(self):
        """ Returns the particle density. """
        return 0.5 * (1 + self.X)

    def evaluate(self, t):
        """
        Calculates the synchrotron frequency at time `t`
        for a shock's movement that is described by `evo`.

        To prevent overflow exceptions and generally slow
        calculations, I take the sum of the log of all powers
        of ten rather than evaluating each separately. The
        model parameters span many, many orders of magnitude.

        To improve efficiency, constant factors are evaluated
        and combined beforehand and the result is used.

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.array of float
            The cooling frequency at time `t` measured in Hz.
        """
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # return synchrotron frequency [Hz]
        return (
            # all constants evaluated
            4.049782158231e+16 *

            # k-dependent factors
            (self.alpha ** -0.5) *
            (self.beta ** -1.5) *

            # model parameters
            (self.n_p ** -2) *      # particle density
            (self.eps_e ** 2) *     # electric field fraction
            (self.eps_b ** 0.5) *   # magnetic field fraction
            ((1 + self.z) ** 0.5) * # redshift
            (self.E ** 0.5) *       # explosion energy
            ((self.p - 2) ** 2) *   # electron energy index
            ((self.p - 1) ** -2) *  # electron energy index
            (t ** -1.5)             # time in days
        )


class AbsorptionFrequencyModel(BaseSpectralModel):
    """
    Absorption frequency model. Assumes an ultra-relativistic
    shock moving through an external medium with rho = rho0
    * R^-k density.

    Attributes
    ----------
    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons, unit=None.

    X : float
        The hydrogen mass fraction, unit=None.

    p : float
        The electron energy power-law index, unit=None.
    """

    # noinspection PyPep8Naming
    def __init__(self, E, rho0, eps_e, eps_b, k, z, X, p):
        super().__init__(E, eps_b, k, z)
        self.eps_e = eps_e
        self.rho0 = rho0
        self.X = X
        self.p = p

    def evaluate(self, t, regime):
        """
        ??

        Parameters
        ----------
        t : float or np.array of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        regime : str, {'slow', 'fast'}
            Indicates whether to evaluate the fast or slow
            cooling model.

        Returns
        -------
        float or np.array of float
            The self-absorption frequency at time `t` measured in Hz.
        """
        return getattr(self, f'evaluate_{regime}')(t)

    def evaluate_slow(self, t):
        """"""
        if isinstance(t, u.Quantity):
            t = t.to_value('d')

        # convenience variables
        k, x = self.k, 4 - self.k
        c = const.c.cgs.value      # noqa
        m_p = const.m_p.cgs.value  # noqa
        q_e = 4.8032e-10           # [g1/2 cm3/2 s-1]

        # exponents for readability
        e_alpha = -(0.8 * (1 - k) / x)
        e_beta = -(0.6*k / x)
        e_pi = (0.2 * (4 + 2*k) / x)
        e_c = -0.8 * ((5 - 2*k) / x)
        e_rho = (2.4 / x)
        e_en = (0.8 * (1 - k) / x)
        e_z = -(0.8 * (5 - 2*k) / x)

        # return self-absorption frequency [Hz]
        return 10 ** (

            # Dimension-less quantities
            np.log10(2 * 3 ** 0.8) +
            e_alpha * np.log10(self.alpha) +        # hydrodynamics coefficient
            e_beta * np.log10(self.beta) +          # hydrodynamics coefficient
            e_pi * np.log10(np.pi) +                # pi
            (1.6 * np.log10(0.5 * (1 + self.X))) +  # hydrogen mass fraction
            -np.log10(self.p - 2) +                 # electron energy index
            1.6 * np.log10(self.p - 1) +            # electron energy index
            -0.6 * np.log10(self.p + 2/3) +         # electron energy index
            0.6 * np.log10(self.p + 2) +            # electron energy index

            # Dimensional quantities
            1.6 * np.log10(q_e) +   # electron charge [g1/2 cm3/2 s-1]
            -1.6 * np.log10(m_p) +  # proton mass [g]
            e_c * np.log10(c) +     # speed of light [cm s-1]

            # Model parameters
            -np.log10(self.eps_e) +                       # electron field energy fraction
            0.2 * np.log10(self.eps_b) +                  # magnetic field electron fraction
            e_rho * (17*k + np.log10(m_p * self.rho0)) +  # density [g cm-3]
            e_en * (52 + np.log10(self.E)) +              # energy [erg]
            e_z * np.log10(1 + self.z) +                  # redshift

            # Evaluated at time, `t`
            -0.6 * (k / x) * np.log10(86_400 * t)  # time [s]
        )

    def evaluate_fast(self, t):
        """"""
        raise NotImplementedError('Not yet implemented.')


class ObservedFluxModel:
    """
    Container for computing the observed afterglow flux.

    Parameters
    ----------
    afterglow_model :
        The afterglow flux model to use. Can be any custom
        defined model as long as it has a `model` method
        that takes an `Observation` and returns an array.

    extinction_model :
        The dust extinction model to use. Models from
        `dust_extinction` package or any custom object
        that has an `extinguish` method.

    ext_sf : np.array, optional
        The pre-computed source frame extinction values.

    ext_mw : np.array, optional
        The pre-computed milky way extinction values.
    """
    def __init__(self, afterglow_model, extinction_model, ext_sf=None, ext_mw=None):
        self.afterglow_model = afterglow_model
        self.extinction_model = extinction_model
        self.ext_sf = ext_sf
        self.ext_mw = ext_mw

    def __repr__(self):
        """ Human-readable representation. """
        return (
            f'ObservedFluxModel('
            f'ag={self.afterglow_model}, '
            f'ext={self.extinction_model})'
        )

    def __call__(self, *args, **kwargs):
        """ Calls the `model` method. """
        return self.model(*args, **kwargs)

    def model(self, obs, params, **kwargs):
        """
        Models the observed GRB afterglow flux.

        Parameters
        ----------
        obs : Observation
            The `Observation` object to model.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        kwargs : dict, optional
            Any additional arguments needed to instantiate the
            flux model.

        Returns
        -------
        np.ndarray of float
            The modeled observed GRB afterglow flux.
        """

        # Model the GRB afterglow flux
        modeled = self.model_afterglow(obs, params, **kwargs)

        # Apply dust extinction and host galaxy corrections
        modeled[obs.sflux_loc] = self.model_extinction(
            modeled[obs.sflux_loc], **Parameters.extrinsic(obs, params)
        )

        return modeled

    def model_afterglow(self, obs, params, **kwargs):
        """
        Models the GRB afterglow flux.

        Parameters
        ----------
        obs : Observation
            The `Observation` object to model.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        kwargs : optional
            Any additional arguments needed to instantiate the
            flux model.

        Returns
        -------
        np.ndarray of float
            The modeled GRB afterglow flux.
        """
        if params.get('shared') is not None:
            modeled = np.full(obs.length, np.nan, dtype=float)

            # Model each data group separately
            for group in obs.data_groups.keys():
                model_params = params.get(group).get('model')

                modeled[obs.data_groups[group]] = self.afterglow_model(
                    **model_params, **kwargs).model(obs, group)

        else:
            # No data groups, model all together
            model = self.afterglow_model(**params.get('model'), **kwargs)
            modeled = model.model(obs)

        # return GRB afterglow flux
        return modeled

    def model_extinction(
        self, modeled, wn, z=None, ebv_sf=None,
        ebv_mw=None, host_pos=None, host_vals=None
    ):
        """
        Corrects the intrinsic flux, `modeled`, for
        dust extinction and host galaxy contributions.

        Parameters
        ----------
        modeled : np.array
            The modeled flux.

        wn : np.array
            The observed wave numbers measured in inverse
            microns.

        z : float, optional
            The redshift. If provided, transforms `wn` to
            the source frame when extinguishing for source
            frame dust.

        ebv_sf : float, optional
            The E(B - V) value for the source frame. The
            precomputed source frame extinction values are
            given priority over `ebv_sf` (if defined).

        ebv_mw : float, optional
            The E(B - V) value for the Milky Way. The
            precomputed source frame extinction values are
            given priority over `ebv_mw` (if defined).

        host_pos, host_vals : dict, optional
            The positions and values of the host galaxy corrections.
            Both must be provided to apply host galaxy corrections.
            Assumes that the values are measured in the same space
            as the intrinsic flux.

        Returns
        -------
        np.ndarray of float
            The extinguished and host galaxy corrected flux.
        """

        # Apply source frame extinction
        if self.ext_sf is not None or ebv_sf is not None:
            modeled *= self.ext_sf if self.ext_sf is not None \
                else self.extinction_model.extinguish((1+z)*wn, Ebv=ebv_sf)

        # Apply host galaxy correction
        if host_vals is not None and host_pos is not None:
            for name, corr in host_vals.items():
                modeled[np.where(host_pos[name])] += corr

        # Apply Milky Way extinction
        if self.ext_mw is not None or ebv_mw is not None:
            modeled *= self.ext_mw if self.ext_mw is not None \
                else self.extinction_model.extinguish(wn, Ebv=ebv_mw)

        # return (afterglow_flux * ext_sf + host_correction) * ext_mw
        return modeled
