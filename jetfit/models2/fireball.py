import math

import astropy.units as u
import astropy.constants as const
import numpy as np
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

from jetfit.core.defns.enums import DataType
from jetfit.core.input import Observation
from jetfit.core.values import IntegratedFlux, SpectralFlux, SpectralIndex
from jetfit.models2.basemodels import SynchrotronFrequencyModel, SpectralFluxModel, IntegratedFluxModel, \
    SpectralIndexModel
from jetfit.models2.basemodels import CoolingFrequencyModel, PeakFluxModel
from jetfit.core.core import two_point_approx

import warnings
warnings.filterwarnings('ignore', category=UserWarning)


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
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X, ebv_mw=None, ebv_sf=None, tj=None, sj=None):
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

        # Jet break props
        self.tj = tj
        self.sj = sj

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
        Sets the density normalization as a simple float.

        Define rho as:

        rho = rho_x * R^-k = rho_0 * (R/R_0)^-k

        such that:

        rho_x = rho_0 * R_0^k

        where R_0 is the characteristic radius which we take to
        be 1e17 cm. Then `rho0` is the reference density at 1e17
        cm with units of g/cm^3. However, I normalize to the proton
        mass such that rho0 is a number density with units of cm^-3.

        Parameters
        ----------
        rho0 : float or astropy.units.Quantity
            The density normalization. If a float or unit-less
            quantity is provided, assumes that the value is
            ??.
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

    def model(
            self,
            observation: Observation,
            cal_offsets: dict = None,
            host_corrs: dict = None
    ) -> np.ndarray:
        """
        Models the observational data.

        Parameters
        ----------
        observation : Observation
            The `Observation` object to model.

        cal_offsets : dict
            Key, value of name, offset value for the cal group.

        host_corrs : dict
            Key, value of name, correction value for the host corr.

        Returns
        -------
        np.ndarray of ??
            The modeled observational data.
        """
        if self.tj is not None:
            return self.model_jet(observation, cal_offsets, host_corrs)

        res = np.full(len(observation.data), np.nan)

        for i, data in enumerate(observation.data):

            # Critical spectral values
            f_peak = self.f_peak(data.time)
            nu_m = self.nu_m(data.time)
            nu_c = self.nu_c(data.time)

            if data.type == DataType.SPECTRAL_FLUX:
                res[i] = SpectralFluxModel(
                    nu_m, nu_c, f_peak, self.p, self.k
                ).model_smooth(data)

            elif data.type == DataType.INTEGRATED_FLUX:
                res[i] = IntegratedFluxModel(
                    nu_m, nu_c, f_peak, self.p, self.k
                ).model_smooth(data)

            elif data.type == DataType.SPECTRAL_INDEX:
                res[i] = SpectralIndexModel(
                    nu_m, nu_c, f_peak, self.p, self.k
                ).model(data)

            if res[i] == np.nan:
                return res

        # Apply extinction to spectral flux values
        if self.ebv_mw or self.ebv_sf:
            mask = observation.spectral_flux_loc
            wn = observation.wave_number_array[mask]

            if self.ebv_mw:  # milky way
                res[mask] *= self.ext_model.extinguish(wn, Ebv=self.ebv_mw)

            if self.ebv_sf:  # source frame
                res[mask] *= self.ext_model.extinguish((1 + self.z) * wn, Ebv=self.ebv_sf)

        # Apply calibration offsets
        if cal_offsets is not None:
            for name, offset in cal_offsets.items():
                res[observation.cal_offsets[name]] *= 10.0 ** -(0.4 * offset)

        # Apply host galaxy correction
        if host_corrs is not None:
            for name, corr in host_corrs.items():
                res[observation.host_corr[name]] += corr

        return res

    def model_jet(
            self,
            observation: Observation,
            cal_offsets: dict = None,
            host_corrs: dict = None
    ) -> np.ndarray:
        """"""
        res = np.full(len(observation.data), np.nan)

        for i, data in enumerate(observation.data):
            # Critical spectral values
            f_peak, f_peak_j = self.f_peak(data.time), self.f_peak(self.tj)
            nu_m, nu_m_j = self.nu_m(data.time), self.nu_m(self.tj)
            nu_c, nu_c_j = self.nu_c(data.time), self.nu_c(self.tj)

            # Model the fluxes
            if data.type != DataType.SPECTRAL_INDEX:

                if data.type == DataType.SPECTRAL_FLUX:
                    x = SpectralFluxModel(nu_m, nu_c, f_peak, self.p, self.k).model_smooth(data)
                    y = SpectralFluxModel(nu_m_j, nu_c_j, f_peak_j, self.p, self.k).model_smooth(data)

                else:
                    x = IntegratedFluxModel(nu_m, nu_c, f_peak, self.p, self.k).model_smooth(data)
                    y = IntegratedFluxModel(nu_m_j, nu_c_j, f_peak_j, self.p, self.k).model_smooth(data)

                res[i] = (x ** (-self.sj) + (y * (data.time.to_value('d') / self.tj) ** -self.p) ** -self.sj) ** -(1 / self.sj)

            else:  # Model the spectral index
                res[i] = SpectralIndexModel(nu_m, nu_c, f_peak, self.p, self.k).model(data)

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

        # Apply calibration offsets
        if cal_offsets is not None:
            for name, offset in cal_offsets.items():
                res[observation.cal_offsets[name]] *= 10.0 ** -(0.4 * offset)

        # Apply host galaxy correction
        if host_corrs is not None:
            for name, corr in host_corrs.items():
                res[observation.host_corr[name]] += corr

        return res

    def f_peak(self, t: u.Quantity | float, evo: str = 'adiabatic'):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : astropy.units.Quantity ot float
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution type.

        Returns
        -------
        ??
            The peak flux in mJy at time `t`.
        """
        return PeakFluxModel(
            self.E, self.rho0, self.eps_b, self.dL, self.z, self.k, self.X)(t)

    def nu_c(self, t: u.Quantity | float, evo: str = 'adiabatic'):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : astropy.units.Quantity or float
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution type.

        Returns
        -------
        ??
            The cooling frequency in Hz at time `t`.
        """
        return CoolingFrequencyModel(
            self.E, self.rho0, self.eps_b, self.k, self.z)(t)

    def nu_m(self, t: u.Quantity | float, evo: str = 'adiabatic'):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : astropy.units.Quantity or float
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        evo : str, {'adiabatic', 'radiative'}, default='adiabatic'
            The evolution type.

        Returns
        -------
        ??
            The synchrotron frequency in Hz at time `t`.
        """
        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p)(t)


class ISMModel(FireballModel):
    """"""
    pass


class WindModel(FireballModel):
    """"""
    pass


if __name__ == '__main__':
    m_p = const.m_p.cgs  # Proton mass [g]
    m_e = const.m_e.cgs  # Election mass [g]
    q_e = u.Quantity(4.8032e-10 * u.g**0.5 * u.cm**1.5 / u.s)  # Electron charge [g1/2 cm3/2 s-1]
    c = const.c.cgs  # Speed of light [cm / s]

    # Check derivation
    A = (4/3) * np.sqrt(2*math.pi) * (q_e**3) * (m_e**-1) * 17 * ((4*math.pi)**-.75) * ((1024*math.pi)**-0.25)

    def vdh_flux(X, n, dL, E, eps_b):
        F = A
        F *= (1 + X) / 2
        F *= m_p ** -0.5
        F *= c ** -3
        F *= n ** 0.5
        F *= eps_b ** 0.5
        F *= dL ** -2
        F *= E
        return F

    # F_vdh_mjy = vdh_flux(0.7, 1 / (u.cm ** 3), 1e28 * u.cm, 1e52 * u.erg, 0.1).to('mJy')
    F_sp_mjy = vdh_flux(1.0, 1 / (u.cm ** 3), 1e28 * u.cm, 1e52 * u.erg, 1.0).to('mJy')

    rhoW = 5e11 / m_p.cgs.value

    # <editor-fold desc="TEST VDH">
    # Test VDH values for k = 0
    vdh = FireballModel(1,2.2,.1,.1,0.0,1., 1.,0.0, 0.7)

    if not math.isclose(vdh.f_peak(1 * u.d) / 0.5, 21.3, abs_tol=0.1):
        raise ValueError('Peak Flux does not match Van Der Horst value.')

    if not math.isclose(vdh.nu_c(1 * u.d) / (0.5**-0.5), 5.98e13, abs_tol=1e12):
        raise ValueError('Cooling Frequency does not match Van Der Horst value.')

    if not math.isclose(vdh.nu_m(1 * u.d) / (0.5**0.5), 8.98e11, abs_tol=1e10):
        raise ValueError('Synchrotron Frequency does not match Van Der Horst value.')
    # </editor-fold>

    # Test SP values for k = 0
    ism = FireballModel(1.,2.5,1,1,0.0,1., 10.,0.0, 1.0)

    nus = np.logspace(12, 18, 100)
    density_model = SpectralFluxModel(nu_m=10**13, nu_c=10**15, f_peak=10**4.5, p=2.5, k=2.0)

    fluxes, fluxes2 = [], []
    for nu in nus:
        fluxes.append(density_model.evaluate_smooth(nu))
        fluxes2.append(density_model.evaluate(nu))
    fluxes = np.array(fluxes)
    fluxes2 = np.array(fluxes2)

    plt.loglog(nus, fluxes)
    plt.loglog(nus, fluxes2, color='black')

    # plt.axvline(x=10**11.1, color='black', linestyle=':', alpha=0.3)
    # plt.axvline(x=10**12.5, color='black', linestyle=':', alpha=0.3)
    plt.axhline(y=10**4.5, color='black', linestyle=':', alpha=0.3)
    plt.show()

    # plt.loglog(nus, np.abs(fluxes - fluxes2), color='red')
    # plt.show()

    if not math.isclose(ism.nu_c(1 * u.d), 2.7e12, abs_tol=1e11):
        raise ValueError(
            f'Cooling Frequency does not match Sari piran value: '
            f'{round(ism.nu_c(1 * u.d) / 1e12, 3)}e12 Hz != 2.7e12 Hz'
        )

    if not math.isclose(ism.nu_m(1 * u.d), 5.7e14, abs_tol=1e13):
        raise ValueError(
            'Synchrotron Frequency does not match Sari piran value: '
            f'{ism.nu_m(1 * u.d)} Hz != 5.7e14 Hz'
        )

    if not math.isclose(ism.f_peak(1 * u.d), 110, abs_tol=15):
        raise ValueError(
            'Peak Flux does not match Sari piran value: '
            f'{ism.f_peak(1 * u.d)} mJy != 110 mJy'
        )

    print()
