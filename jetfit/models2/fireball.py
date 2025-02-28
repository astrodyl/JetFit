import astropy.units as u
import astropy.constants as const
import numpy as np
from dust_extinction.parameter_averages import CCM89

from jetfit.core.defns.enums import DataType
from jetfit.core.input import Observation
from jetfit.core.values import IntegratedFlux, SpectralFlux, SpectralIndex
from jetfit.models2.basemodels import SynchrotronFrequencyModel, SpectralFluxModel, IntegratedFluxModel
from jetfit.models2.basemodels import CoolingFrequencyModel, PeakFluxModel
from jetfit.core.core import two_point_approx

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define constants in useful units
m_p = const.m_p.cgs # Mass of proton [g]


class FireballModel:
    """
    Implements the ultra-relativistic shock moving into an
    external medium with density rho = rho_0 * R^-k.

    Parameters can be passed either as an Astropy Quantity with
    an associated unit, or as a simple float. However, they will
    only be stored as floats. Quantities will be converted to the
    appropriate units before storing the float value. This is done
    because operating with Astropy Quantities is very slow and makes
    MCMC fitting extremely difficult. If a float is passed, it is
    assumed that the value is already in the expected units.

    Attributes
    ----------
    E : float or astropy.units.Quantity
        The explosion energy normalized to 1e52 ergs.

    p : float
        The electron energy index (dimensionless).

    eps_b : float
        The fraction of thermal energy in the magnetic field.

    eps_e : float
        The fraction of thermal energy carried by relativistic
        electrons.

    z : float
        The redshift to the event.

    dL : float or astropy.units.Quantity
        The luminosity distance to the event.

    rho0 : float or astropy.units.Quantity
        The density normalization.

    k : float
        The density power-law index.

    X : float
        The hydrogen mass fraction.

    References
    ----------
    [1] Broadband view of blast wave physics: A study
        of gamma-ray burst afterglows
    """

    # noinspection PyPep8Naming
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X, ebv_mw=None, ebv_sf=None):
        # intrinsic properties
        self.E = E
        self.p = p
        self.eps_b = eps_b
        self.eps_e = eps_e
        self.k = k
        self.rho0 = rho0
        self.X = X

        # extrinsic properties
        self.dL = dL
        self.z = z

        # temp
        self.ebv_mw = ebv_mw
        self.ebv_sf = ebv_sf
        self.ext_model = CCM89(Rv=3.1)

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """ Returns the explosion energy normalized to 10e52 ergs. """
        return self._E

    # noinspection PyPep8Naming
    @E.setter
    def E(self, e: float | u.Quantity) -> None:
        """
        Sets the explosion energy normalized to 10e52 ergs.

        Parameters
        ----------
        e : float or astropy.units.Quantity
            The explosion energy. If a float is provided, assumes
            that the value is already normalized to 1e52 ergs.
        """
        if isinstance(e, u.Quantity):
            e = e.to_value('erg') / 1e52

        self._E = e

    @property
    def rho0(self) -> float:
        """ Returns the density normalization, normalized to the proton mass. """
        return self._rho0

    @rho0.setter
    def rho0(self, rho0) -> None:
        """
        Sets the density normalization as an astropy Quantity
        with units of g * cm^k-3. If a float or unit-less
        quantity is provided, assumes that the value is
        normalized to m_p [g / cm^3-k].

        Define rho0 as:

        rho0 = rho_x * r_x ^ k-3

        where rho_x is the reference density at the characteristic
        radius, r_x. We choose r_x = 1 cm such that:

        rho0 = n_x * m_p * 1cm ^ k-3

        where n_x is the reference number density at r_x = 1cm and
        m_p is the mass of a proton.

        Parameters
        ----------
        rho0 : float or astropy.units.Quantity
            The density normalization. If a float or unit-less
            quantity is provided, assumes that the value is
            normalized to the mass of the proton (g) * cm^3-k.
        """
        if isinstance(rho0, u.Quantity):
            rho0 = rho0.cgs.value

        self._rho0 = rho0

    # noinspection PyPep8Naming
    @property
    def dL(self) -> float:
        """ Returns the luminosity distance normalized to 1e28 cm. """
        return self._dL

    # noinspection PyPep8Naming
    @dL.setter
    def dL(self, d: float | u.Quantity) -> None:
        """
        Sets the luminosity distance normalized to 1e28 cm.

        Parameters
        ----------
        d : float or astropy.units.Quantity
            The luminosity distance. If a float is provided,
            assumes the value is already normalized to 1e28 cm.
        """
        if isinstance(d, u.Quantity):
            d = d.to_value('cm') / 1e28

        self._dL = d

    def model(self, observation: Observation) -> np.ndarray:
        """
        Models the observational data.

        Parameters
        ----------
        observation : Observation
            The `Observation` object to model.

        Returns
        -------
        np.ndarray
            The modeled observational data.
        """
        res = np.full(len(observation.data), np.nan)

        for i, data in enumerate(observation.data):

            if data.type != DataType.SPECTRAL_INDEX:
                res[i] = self.model_flux(data)

            elif data.type == DataType.SPECTRAL_INDEX:
                res[i] = self.model_index(data)

            if res[i] == np.nan:
                return res

        # Apply extinction to spectral flux values
        if self.ebv_mw or self.ebv_sf:
            mask = observation.flux_types == DataType.SPECTRAL_FLUX
            wn = observation.wave_number_array[mask]

            if self.ebv_mw:  # milky way
                res[mask] *= self.ext_model.extinguish(wn, Ebv=self.ebv_mw)

            if self.ebv_sf:  # source frame
                res[mask] *= self.ext_model.extinguish((1 + self.z) * wn, Ebv=self.ebv_sf)

        return res

    def model_flux(self, data):
        """
        Models an observational flux value.

        Parameters
        ----------
        data : SpectralFlux or IntegratedFlux
            The value to model.

        Returns
        -------
        u.Quantity
            The modeled flux.
        """
        f_peak, nu_c, nu_m = (
            self.f_peak(data.time),
            self.nu_c(data.time),
            self.nu_m(data.time)
        )

        if data.type == DataType.SPECTRAL_FLUX:
            return SpectralFluxModel(nu_m, nu_c, f_peak, self.p).model(data)

        return IntegratedFluxModel(nu_m, nu_c, f_peak, self.p).model(data)

    def model_index(self, val):
        """
        Approximates the spectral index measurement using
        `val`'s integration range and time range.

        Parameters
        ----------
        val : SpectralIndex
            The value to model.

        Returns
        -------
        u.Quantity
            The modeled flux.
        """
        f_peak, nu_c, nu_m = (
            self.f_peak(val.time_range.lower),
            self.nu_c(val.time_range.lower),
            self.nu_m(val.time_range.lower)
        )

        f_peak2, nu_c2, nu_m2 = (
            self.f_peak(val.time_range.upper),
            self.nu_c(val.time_range.upper),
            self.nu_m(val.time_range.upper)
        )

        # Use the critical values to initialize the models
        f_start_model = IntegratedFluxModel(nu_m, nu_c, f_peak, self.p)
        f_stop_model = IntegratedFluxModel(nu_m2, nu_c2, f_peak2, self.p)

        # Evaluate the models
        f_start = f_start_model(val.int_range.lower.value, val.int_range.upper.value)
        f_stop = f_stop_model(val.int_range.lower.value, val.int_range.upper.value)

        # Return the approximated spectral index
        return two_point_approx(
            f_stop, f_start, val.int_range.lower.value, val.int_range.upper.value, log=True
        )

    def f_peak(self, t: u.Quantity, evo: str = 'adiabatic'):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : astropy.units.Quantity
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution type.

        Returns
        -------
        astropy.units.Quantity
            The peak flux in mJy at time `t`.
        """
        return PeakFluxModel(self.E, self.rho0, self.eps_b, self.dL,self.z, self.k, self.X)(t, evo)

    def nu_c(self, t, evo: str = 'adiabatic'):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : astropy.units.Quantity
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution type.

        Returns
        -------
        astropy.units.Quantity
            The cooling frequency in Hz at time `t`.
        """
        return CoolingFrequencyModel(self.E, self.rho0, self.eps_b, self.k, self.z)(t, evo)

    def nu_m(self, t: u.Quantity, evo: str = 'adiabatic'):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : astropy.units.Quantity
            The time to evaluate.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution type.

        Returns
        -------
        astropy.units.Quantity
            The synchrotron frequency in Hz at time `t`.
        """
        return SynchrotronFrequencyModel(self.E, self.rho0, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p)(t, evo)


class ISMModel(FireballModel):
    """"""
    pass


class WindModel(FireballModel):
    """"""
    pass


if __name__ == '__main__':
    rhoC = 1
    rhoW = 5e11 / m_p.cgs.value

    v = FireballModel(1,2.2,.1,.1,0.0,1., rhoC,0.0, 0.7)

    test_f_peak = v.f_peak(1 * u.d)
    test_nu_c = v.nu_c(1 * u.d)
    test_nu_m = v.nu_m(1 * u.d)

    print()
