

class ModelParameters:
    """
    Model Parameters for JetFit.

    Attributes
    ----------
    energy : float
        Explosion energy normalized to ???

    asymptotic_lorentz : float
        Asymptotic Lorentz factor for the Boosted Fireball model (eta0).

    density : float
        Uniform circumburst density normalized to ??

    z : float
        Redshift to the GRB.

    boost : float
        Lorentz boost factor the Boosted Fireball model (GammaB).

    obs_angle : float
        Observation (Viewing) angle measured in radians.

    distance : float
        Luminosity distance to the GRB normalized to 10^28 cm.

    electron index : float
        Electron energy index

    accel_electron_fraction : float
        Fraction of electrons accelerated by the blast

    electron_energy_fraction : float
        ??

    magnetic_energy_fraction : float
        ??
    """
    def __init__(self, explosion_energy: float, asymptotic_lorentz_factor: float,
                 circumburst_density: float, redshift: float, boost_lorentz_factor: float,
                 obs_angle: float, luminosity_distance: float, electron_energy_index: float,
                 accelerated_electron_fraction: float, electron_energy_fraction: float,
                 magnetic_energy_fraction: float):

        # Hydrodynamic Parameters
        self.energy = explosion_energy
        self.boost = boost_lorentz_factor
        self.density = circumburst_density
        self.asymptotic_lorentz = asymptotic_lorentz_factor

        # Radiation Parameters
        self.electron_index = electron_energy_index
        self.electron_energy_fraction = electron_energy_fraction
        self.magnetic_energy_fraction = magnetic_energy_fraction
        self.accel_electron_fraction = accelerated_electron_fraction

        # Observational Parameters
        self.z = redshift
        self.obs_angle = obs_angle
        self.distance = luminosity_distance
