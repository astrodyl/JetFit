import numpy as np

from jetfit.core.defns.enums import ScaleType
from jetfit.core.utils import maths
from jetfit.mcmc.parameters.parameters import MCMCFixedParameter
from jetfit.mcmc.parameters.parameters import MCMCFittingParameter


class BFModelParams:
    """
    Boosted Fireball Model Parameters.

    Attributes
    ----------
    explosion_energy : float
        The normalized Explosion energy denoted by :math:`E_{0, 50}`.

    asymptotic_lorentz_factor : float
        The asymptotic Lorentz factor denoted by :math:`\eta_{0}`.

    circumburst_density : float
        The uniform circumburst density denoted by :math:`n_`.

    redshift : float
        The redshift to the GRB.

    boost_lorentz_factor : float
        The lorentz boost factor the Boosted Fireball model (GammaB).

    obs_angle : float
        The observation (Viewing) angle measured in radians.

    luminosity_distance : float
        Luminosity distance to the GRB normalized to 10^28 cm.

    electron index : float
        The electron energy index.

    accelerated_electron_fraction : float
        The fraction of electrons accelerated by the blast.

    electron_energy_fraction : float
        The fraction of energy transferred to the electric field.

    magnetic_energy_fraction : float
        The fraction of energy transferred to the magnetic field.

    ebv_milky_way : float
        The milky way E(B - V) strength.

    ebv_source_frame : float
        The source frame E(B - V) strength.
    """
    def __init__(
            self,
            explosion_energy: float = None,
            asymptotic_lorentz_factor: float = None,
            circumburst_density: float = None,
            redshift: float = None,
            boost_lorentz_factor: float = None,
            obs_angle: float = None,
            luminosity_distance: float = None,
            electron_energy_index: float = None,
            accelerated_electron_fraction: float = None,
            electron_energy_fraction: float = None,
            magnetic_energy_fraction: float = None,
            ebv_milky_way: float = None,
            ebv_source_frame: float = None
    ):

        # Hydrodynamic Parameters
        self.explosion_energy = explosion_energy
        self.circumburst_density = circumburst_density
        self.boost_lorentz_factor = boost_lorentz_factor
        self.asymptotic_lorentz_factor = asymptotic_lorentz_factor

        # Radiation Parameters
        self.electron_energy_index = electron_energy_index
        self.accelerated_electron_fraction = accelerated_electron_fraction
        self.electron_energy_fraction = electron_energy_fraction
        self.magnetic_energy_fraction = magnetic_energy_fraction

        # Observational Parameters
        self.ebv_milky_way = ebv_milky_way
        self.ebv_source_frame = ebv_source_frame
        self.redshift = redshift
        self.obs_angle = obs_angle
        self.luminosity_distance = luminosity_distance

    # <editor-fold desc="Mathematical Representations">
    @property
    def p(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        electron energy index to make equations less verbose.

        Returns
        -------
        float
            The electron energy index.
        """
        return self.electron_energy_index

    # noinspection PyPep8Naming
    @property
    def E(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        explosion energy to make equations less verbose.

        Returns
        -------
        float
            The explosion energy.
        """
        return self.explosion_energy

    @property
    def n(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        circumburst density to make equations less verbose.

        Returns
        -------
        float
            The circumburst density.
        """
        return self.circumburst_density

    @property
    def eta(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        asymptotic Lorentz factor to make equations less verbose.

        Returns
        -------
        float
            The asymptotic Lorentz factor.
        """
        return self.asymptotic_lorentz_factor

    @property
    def gamma_b(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        boost Lorentz factor to make equations less verbose.

        Returns
        -------
        float
            The boost Lorentz factor.
        """
        return self.boost_lorentz_factor

    @property
    def theta(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        observing angle to make equations less verbose.

        Returns
        -------
        float
            The observing angle measured in radians.
        """
        return self.obs_angle

    @property
    def zeta(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        accelerated electron fraction to make equations less verbose.

        Returns
        -------
        float
            The accelerated electron fraction.
        """
        return self.accelerated_electron_fraction

    @property
    def eps_e(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        electron energy fraction to make equations less verbose.

        Returns
        -------
        float
            The electron energy fraction.
        """
        return self.electron_energy_fraction

    @property
    def eps_b(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        magnetic energy fraction to make equations less verbose.

        Returns
        -------
        float
            The magnetic energy fraction.
        """
        return self.magnetic_energy_fraction

    @property
    def d(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        luminosity distance to make equations less verbose.

        Returns
        -------
        float
            The luminosity distance normalized to 10e28 cm.
        """
        return self.luminosity_distance

    @property
    def z(self) -> float:
        """
        Maps the mathematical representation to the descriptive value for the
        redshift to make equations less verbose.

        Returns
        -------
        float
            The redshift.
        """
        return self.redshift
    # </editor-fold>

    @property
    def missing(self) -> list:
        """
        Determines which model parameters are ``None``.

        Returns
        -------
        list of str
            The names of parameters that are ``None``.
        """
        return [k for k, v in vars(self).items() if v is None]

    @classmethod
    def from_dict(cls, d: dict):
        """
        Instantiates the BFModelParams from a dictionary.

        Non-class attributes will be ignored.

        Parameters
        ----------
        d : dict
            Dictionary of parameter values.

        Returns
        -------
        BFModelParams
            The Boosted Fireball model parameters in linear scale.
        """
        return cls(**{k : v for k, v in d.items() if hasattr(cls(), k)})

    @classmethod
    def from_mcmc_samples(
            cls,
            fixed: list[MCMCFixedParameter],
            fitting: list[MCMCFittingParameter],
            theta: np.ndarray[float]
    ):
        """
        Instantiates the BFModelParams from MCMC samples.

        Since the `emcee` package returns samples in a numpy array, there
        needs to be some record of the order of the array in order to map
        the sample array to the model parameters.

        The MCMC class does this by storing a list of the fitting parameters
        in the `fitting` attribute. The order of `fitting` is guaranteed to
        be the same order as the `emcee` sample array since it was created
        from the `fitting` attribute.

        Since the Boosted Fireball model needs to be evaluated in linear scale,
        all parameters are first converted to linear scale before setting them.

        Parameters
        ----------
        fixed : list of MCMCFixedParameter
            The fixed model parameters.

        fitting : list of MCMCFittingParameter
            The fitting (free) model parameters.

        theta : np.ndarray of float
            The MCMC sampled parameters.

        Returns
        -------
        BFModelParams
            The Boosted Fireball model parameters in linear scale.
        """
        params = {}

        for i, p in enumerate(fitting):
            params[p.name] = maths.to_scale(
                theta[i], p.scale, ScaleType.LINEAR
            )

        for i, p in enumerate(fixed):
            params[p.name] = maths.to_scale(
                p.value, p.scale, ScaleType.LINEAR
            )

        return cls.from_dict(params)
