import math

import astropy.units as u
import astropy.constants as const
import numpy as np
from dust_extinction.parameter_averages import CCM89
from matplotlib import pyplot as plt

from jetfit.core.input import Observation
from jetfit.models2.basemodels import IntegratedFluxModel, SpectralIndexModel
from jetfit.models2.basemodels import SynchrotronFrequencyModel, SpectralFluxModel
from jetfit.models2.basemodels import CoolingFrequencyModel, PeakFluxModel

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
    def __init__(self, E, p, eps_b, eps_e, z, dL, rho0, k, X, tj=None, sj=None):
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
    ) -> np.ndarray:
        """
        Models an observation object.

        Parameters
        ----------
        observation
        cal_offsets

        Returns
        -------
        np.ndarray of float
            The modeled flux.
        """
        res = np.full(len(observation.data), np.nan)

        # For speed, get observation as arrays
        arrays = observation.as_arrays

        # Save locations for readability
        sf_mask = arrays.sflux_loc
        if_mask = arrays.iflux_loc
        si_mask = arrays.sindex_loc

        # Calculate the spectral functions
        f_peaks = self.f_peak(arrays.times)
        nu_ms = self.nu_m(arrays.times)
        nu_cs = self.nu_c(arrays.times)

        # Model spectral fluxes
        res[sf_mask] = SpectralFluxModel(
            nu_ms[sf_mask], nu_cs[sf_mask], f_peaks[sf_mask], self.p, self.k
        ).evaluate(arrays.frequencies[sf_mask])

        # Model Integrated fluxes
        res[if_mask] = IntegratedFluxModel(
            nu_ms[if_mask], nu_cs[if_mask], f_peaks[if_mask], self.p, self.k
        ).evaluate(arrays.if_lower_freqs[if_mask], arrays.if_upper_freqs[if_mask])

        # Model Spectral indices
        res[si_mask] = SpectralIndexModel(
            nu_ms[si_mask], nu_cs[si_mask], f_peaks[si_mask], self.p, self.k
        ).evaluate(arrays.si_lower_freqs[si_mask], arrays.si_upper_freqs[si_mask])

        # Smooth the flux values if there is a jet break
        if self.tj:
            # Smooth the spectral flux
            res[sf_mask] = self.smooth_jet_break(
                f=res[sf_mask],
                t=arrays.times[sf_mask],
                nu=arrays.frequencies[sf_mask]
            )

            # Smooth the integrated flux
            res[if_mask] = self.smooth_jet_break(
                f=res[if_mask],
                t=arrays.times[if_mask],
                lower=arrays.if_lower_freqs[if_mask],
                upper=arrays.if_upper_freqs[if_mask]
            )

        # Apply calibration offsets
        if cal_offsets is not None:
            for name, offset in cal_offsets.items():
                res[observation.cal_offsets[name]] *= 10.0 ** -(0.4 * offset)

        # return modeled observational data
        return res

    def evaluate_spectral_flux(self, t, f):
        """ Model spectral fluxes. """
        res = SpectralFluxModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(f)

        if self.tj:
            return self.smooth_jet_break(res, t, nu=f)

        return res

    def evaluate_integrated_flux(self, t, lower, upper):
        """ Model Integrated fluxes. """
        res = IntegratedFluxModel(
            self.nu_m(t), self.nu_c(t), self.f_peak(t), self.p, self.k
        ).evaluate(lower, upper)

        if self.tj:
            return self.smooth_jet_break(res, t, lower=lower, upper=upper)

        return res

    def smooth_jet_break(self, f, t, **kwargs):
        """
        Smooths the flux across a jet break.

        Parameters
        ----------
        f : np.ndarray or float
            The flux to smooth.

        t : np.ndarray or float
            The time of the flux measurements.

        kwargs :
            Do not pass both `nu` and `lower`/`upper`. OR ELSE.

            nu : np.ndarray or float
                The frequency to evaluate the jet break flux.
                Required if passing spectral fluxes.

            lower : np.ndarray or float
                The lower frequency to evaluate the jet break flux.
                Required if passing integrated fluxes.

            upper : np.ndarray or float
                The upper frequency to evaluate the jet break flux.
                Required if passing integrated fluxes.

        Returns
        -------
        np.ndarray or float
            The jet-break smoothed flux.

        Raises
        ------
        ValueError
            If `nu` and `lower` and `upper` are not provided.
        """
        f_peak_jet = self.f_peak(self.tj)
        nu_m_jet = self.nu_m(self.tj)
        nu_c_jet = self.nu_c(self.tj)

        if 'nu' in kwargs:
            model = SpectralFluxModel
        elif 'lower' in kwargs and 'upper' in kwargs:
            model = IntegratedFluxModel
        else:
            raise ValueError(
                'Must provide either `nu` or `lower` and `upper`'
            )

        jet_model = model(nu_m_jet, nu_c_jet, f_peak_jet, self.p, self.k)
        jet_flux = jet_model(**kwargs)

        return (
            f ** (-self.sj) + (jet_flux * (t / self.tj) ** -self.p) ** -self.sj
        ) ** -(1 / self.sj)

    def f_peak(self, t):
        """
        Calculates the peak flux in the case of an ultra-
        relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The peak flux in mJy at time `t`.
        """
        return PeakFluxModel(
            self.E, self.rho0, self.eps_b, self.dL, self.z, self.k, self.X)(t)

    def nu_c(self, t):
        """
        Calculates the cooling frequency in the case of an ultra-
        relativistic shock moving into an external medium with
        density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The cooling frequency in Hz at time `t`.
        """
        return CoolingFrequencyModel(
            self.E, self.rho0, self.eps_b, self.k, self.z)(t)

    def nu_m(self, t):
        """
        Calculates the synchrotron frequency in the case of an
        ultra-relativistic shock moving into an external medium
        with density rho = rho_0 * R^-k.

        Parameters
        ----------
        t : float or np.ndarray of float or u.Quantity['time']
            The time to evaluate. If `t` is a float, must
            be measured in days since trigger.

        Returns
        -------
        float or np.ndarray of float or u.Quantity['time']
            The synchrotron frequency in Hz at time `t`.
        """
        return SynchrotronFrequencyModel(
            self.E, self.eps_e, self.eps_b, self.k, self.z, self.X, self.p)(t)


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

    nus = np.logspace(10, 18, 100)
    density_model = SpectralFluxModel(nu_m=10**11, nu_c=3*10**12, f_peak=10**1.5, p=2.5, k=0.0)

    fluxes = np.empty(nus.size)
    for i, nu in enumerate(nus):
        fluxes[i] = density_model(nu)

    plt.loglog(nus, fluxes)
    plt.axhline(y=10**1.5, color='black', linestyle=':', alpha=0.3)
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
