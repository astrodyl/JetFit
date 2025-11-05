import numpy as np
from scipy import stats
from typing_extensions import override

from jetfit.core.structs import Prior
from jetfit.core.structs import BoundedMixin


def prior_factory(d: dict):
    """
    Instantiates the appropriate prior class.

    Parameters
    ----------
    d : dict
        Contains the required key : value pairs for
        the prior specified using the `type` key.

    Returns
    -------
    One of valid Prior objects.
        Instantiated from the dict `d`.
    """
    ptype = Prior(d.get('type'))

    match ptype:
        case Prior.GAUSSIAN:
            return GaussianPrior.from_dict(d)

        case Prior.TGAUSSIAN:
            return TruncatedGaussianPrior.from_dict(d)

        case Prior.UNIFORM:
            return UniformPrior.from_dict(d)

        case Prior.SINE:
            return SinePrior.from_dict(d)

        case Prior.MILKYWAYRV:
            return MilkyWayRvPrior.from_dict(d)

        case Prior.POWERLAW:
            return PowerLawPrior.from_dict(d)


class GaussianPrior:
    """
    Gaussian Prior.

    Attributes
    ----------
    mu : float
        The mean of the distribution.

    sigma : float
        The standard deviation.
    """
    type = Prior.GAUSSIAN

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(mu={self.mu}, sigma={self.sigma})"

    @property
    def lower(self) -> float:
        """ Returns the lower 3-sigma bound of the prior. """
        return self.mu - 3.0 * self.sigma

    @property
    def upper(self) -> float:
        """ Returns the upper 3-sigma bound of the prior. """
        return self.mu + 3.0 * self.sigma

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates instance from dict ensuring values are OK.

        Parameters
        ----------
        d : dict
            `mu`    : float
            `sigma` : float

        Returns
        -------
        GaussianPrior
            Instantiated from dictionary
        """
        if (mu := d.get('mu')) is not None:
            if not isinstance(mu, (int, float)):
                raise TypeError('Mu must be a number.')

        if (sigma := d.get('sigma')) is not None:
            if not isinstance(sigma, (int, float)):
                raise TypeError('Sigma must be a number.')

        return cls(mu, sigma)

    def draw(self, n: int) -> float | np.ndarray:
        """
        Draws `n` samples from the Gaussian distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        Returns
        -------
        float or np.ndarray, with shape (n, 1)
            The samples drawn.

        See Also
        --------
        `np.random.normal`

        Examples
        --------
        Draw samples from the normal distribution:

        >>> prior = GaussianPrior(mu=0.3, sigma=0.1)
        >>> s = prior.draw(n=100)

        Verify that the mean is ~= mu:

        >>> prior.mu - np.mean(s)
        """
        return np.random.normal(self.mu, self.sigma, size=n)

    def evaluate(self, x) -> float | np.ndarray:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float or array_like
            The sampled value.

        Returns
        -------
        float
            The prior evaluated at `x`.

        Examples
        --------
        Evaluate the Gaussian prior at `x`:

        >>> prior = GaussianPrior(mu=0.3, sigma=0.1)
        >>> p = prior.evaluate(x)
        """
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)


class MilkyWayRvPrior:
    """
    Milky Way Rv Prior.

    References
    ----------
    Adam Trotter (2011):
        The Gamma-Ray Burst Afterglow Modeling Project:
        Foundational Statistics and Absorption & Extinction Models.
        See: Page 108.
    """
    def __init__(self):
        self.mu = 0.4150
        self.sigma_low = 0.00779
        self.sigma_high = 0.09074

    def __repr__(self) -> str:
        """ Human-readable representation. """
        return (
            f"{self.__class__.__name__}(mu={self.mu}, "
            f"sigma={(self.sigma_low, self.sigma_high)})"
        )

    @property
    def lower(self) -> float:
        """ Returns the lower 3-sigma bound of the prior. """
        return 0.302  # self.mu - 3.0 * self.sigma_low

    @property
    def upper(self) -> float:
        """ Returns the upper 3-sigma bound of the prior. """
        return 0.778  # self.mu + 3.0 * self.sigma_high

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates instance from dict ensuring values are OK.

        Parameters
        ----------
        d : dict
            `mu`   : float
            `low`  : float
            `high` : float

        Returns
        -------
        GaussianPrior
            Instantiated from dictionary
        """
        return cls()

    def draw(self, n: int) -> float | np.ndarray:
        """
        Draws `n` samples from the asymmetric distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        Returns
        -------
        float or np.ndarray, with shape (n, 1)
            The samples drawn.
        """
        # Probabilities for each side (proportional to sigmas)
        p_left = self.sigma_low / (self.sigma_low + self.sigma_high)

        # Randomly decide which branch for each sample
        branches = np.random.rand(n) < p_left

        samples_log = np.zeros(n)

        # Left branch: truncated normal, x < mu
        n_left = branches.sum()
        if n_left > 0:
            samples_log[branches] = stats.truncnorm.rvs(
                -np.inf, 0.0, loc=self.mu, scale=self.sigma_low, size=n_left
            )

        # Right branch: truncated normal, x >= mu
        n_right = n - n_left
        if n_right > 0:
            samples_log[~branches] = stats.truncnorm.rvs(
                0.0, np.inf, loc=self.mu, scale=self.sigma_high, size=n_right
            )

        return samples_log

    def evaluate(self, x) -> float | np.ndarray:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float or array_like
            The sampled value.

        Returns
        -------
        float or np.ndarray of float
            The prior evaluated at `x`.
        """
        sig = self.sigma_low if x < self.mu else self.sigma_high
        norm = (self.sigma_low + self.sigma_high) * np.sqrt(np.pi / 2)

        return np.exp(-0.5 * ((x - self.mu) / sig) ** 2) / norm


class TruncatedGaussianPrior(GaussianPrior, BoundedMixin):
    """
    TruncatedGaussianPrior

    The ``TruncatedGaussianPrior`` is bounded such that the probability is
    evaluated as a ``GaussianPrior`` within the bounds and `-infinity` outside.
    """
    type = Prior.TGAUSSIAN

    def __init__(self, mu: float, sigma: float, lower: float, upper: float):
        BoundedMixin.__init__(self, lower, upper)
        super().__init__(mu, sigma)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (
            f"{class_name}(mu={self.mu}, sigma={self.sigma}, "
            f"lower={self.lower}, upper={self.upper})"
        )

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates instance from dict ensuring values are OK.

        Parameters
        ----------
        d : dict
            `mu`    : float
            `sigma` : float
            `lower  : float
            `upper` : float

        Returns
        -------
        TruncatedGaussianPrior
            Instantiated from dictionary
        """
        valid_types = (int, float)

        if not isinstance(mu := d.get('mu'), valid_types):
            raise TypeError('TGaussian `mu` must be a number.')

        if not isinstance(sigma := d.get('sigma'), valid_types):
            raise TypeError('TGaussian `sigma` must be a number.')

        if not isinstance(lower := d.get('lower'), valid_types):
            raise TypeError('TGaussian lower must be a number.')

        if not isinstance(upper := d.get('upper'), valid_types):
            raise TypeError('TGaussian upper must be a number.')

        return cls(mu, sigma, lower, upper)

    def draw(self, n: int) -> float | np.ndarray:
        """
        Draws ``n`` samples from the truncated Gaussian distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        Returns
        -------
        float or np.ndarray, with shape (n, 1)
            Drawn samples from the parameterized normal distribution.

        Examples
        --------
        Draw samples from the truncated Gaussian distribution:

        >>> p = TruncatedGaussianPrior(mu=0.5, sigma=0.1, lower=0, upper=1)
        >>> s = p.draw(n=100)
        """
        lower = (self.lower - self.mu) / self.sigma
        upper = (self.upper - self.mu) / self.sigma

        return stats.truncnorm(lower, upper, loc=self.mu, scale=self.sigma).rvs(size=n)

    def evaluate(self, x) -> float | np.ndarray:
        """
        Evaluates the truncated Gaussian prior at the value ``x``.

        Parameters
        ----------
        x : float or array_like
            The value to be evaluated.

        Returns
        -------
        float
            The prior evaluated at ``x`` if the sampled values is within the
            `bounds`, else -`np.inf`.

        Examples
        --------
        Evaluate the truncated Gaussian prior at ``x``:

        >>> prior = TruncatedGaussianPrior(mu=0.5, sigma=0.1, lower=0, upper=1)
        >>> s = prior.evaluate(1.1)

        Verify that values out of bounds returns -np.inf:

        >>> prior.evaluate(100)
        -np.inf
        """
        lower = (self.lower - self.mu) / self.sigma
        upper = (self.upper - self.mu) / self.sigma

        return stats.truncnorm.pdf(x, lower, upper, loc=self.mu, scale=self.sigma) \
            if self.encompasses(x) else -np.inf


class UniformPrior(BoundedMixin):
    """
    Uniform prior.

    Attributes
    ----------
    initial_guess : float
        The expected position in the prior.

    initial_sigma : float
        The expected one-sided sigma of the initial position.
    """
    type = Prior.UNIFORM

    def __init__(
        self,
        lower: float,
        upper: float,
        initial_guess: float = None,
        initial_sigma: float = None
    ):
        BoundedMixin.__init__(self, lower, upper)
        self.initial_guess = initial_guess
        self.initial_sigma = initial_sigma

    def __repr__(self) -> str:
        """ Human-readable representation. """
        class_name = self.__class__.__name__
        return f"{class_name}(lower={self.lower}, upper={self.upper})"

    def __call__(self, *args, **kwargs):
        """ Calls self.evaluate(*args, **kwargs) """
        return self.evaluate(*args, **kwargs)

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates instance from dict ensuring values are OK.

        Parameters
        ----------
        d : dict
            Dictionary containing key, value pairs of class parameters.

        Returns
        -------
        UniformPrior
            Instantiated from a dictionary.

        Raises
        ------
        TypeError
        """
        if not isinstance(lower := d.get('lower'), (int, float)):
            raise TypeError('Uniform lower must be a number.')

        if not isinstance(upper := d.get('upper'), (int, float)):
            raise TypeError('Uniform upper must be a number.')

        if (initial := d.get('initial_guess')) is not None:
            if not isinstance(initial, (int, float)):
                raise TypeError('Initial guess must be a number.')

        if (sigma := d.get('initial_sigma')) is not None:
            if not isinstance(sigma, (int, float)):
                raise TypeError('Initial sigma must be a number.')

        return cls(lower, upper, initial, sigma)

    def _bounds(self, initial: bool):
        """
        Returns the bounds of prior considering the initial region.

        Parameters
        ----------
        initial : bool
            If True, and `self.initial_guess` and `self.initial_sigma`
            are set, samples will be drawn from a truncated region
            centered around `initial_guess ± initial_sigma`, clipped
            to within [lower, upper]. If False, samples are drawn from
            the full prior range.

        Returns
        -------
        tuple
        """
        if initial:
            if self.initial_guess is not None and self.initial_sigma is not None:
                return (
                    max(self.initial_guess - self.initial_sigma, self.lower),
                    min(self.initial_guess + self.initial_sigma, self.upper),
                )
        return self.lower, self.upper

    def draw(self, n: int, initial: bool = True) -> float | np.ndarray:
        """
        Draws ``n`` samples from the uniform distribution.

        Draws from `initial_guess` +/- `initial_sigma` if they are both
        defined. Else, draws between ``lower`` and ``upper``.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        initial : bool
            If True, and `self.initial_guess` and `self.initial_sigma`
            are set, samples will be drawn from a truncated region
            centered around `initial_guess ± initial_sigma`, clipped
            to within [lower, upper]. If False, samples are drawn from
            the full prior range.

        Returns
        -------
        np.ndarray or float
            Drawn sample(s) from the uniform distribution.
        """
        return np.random.uniform(*self._bounds(initial), size=n)

    def evaluate(self, x: float) -> float:
        """
        Evaluates the uniform prior at the sampled value `x`.

        Parameters
        ----------
        x : float
            The sampled value.

        Returns
        -------
        float
            `Zero` if `x` is within bounds else ``-np.inf``.
        """
        return 0.0 if self.encompasses(x) else -np.inf


class PowerLawPrior(UniformPrior):
    """
    Uniform prior.

    Attributes
    ----------
    initial_guess : float
        The expected position in the prior.

    initial_sigma : float
        The expected one-sided sigma of the initial position.
    """
    def __init__(
        self,
        lower: float,
        upper: float,
        exponent: int,
        initial_guess: float = None,
        initial_sigma: float = None
    ):
        super().__init__(lower, upper, initial_guess, initial_sigma)
        self.exponent = exponent

    @override
    def draw(self, n: int, initial: bool = True) -> float | np.ndarray:
        """
        Draw samples from the power-law prior distribution.

        Parameters
        ----------
        n : int
            The number of samples to draw.

        initial : bool, optional
            If True, and `self.initial_guess` and `self.initial_sigma`
            are set, samples will be drawn from a truncated region
            centered around `initial_guess ± initial_sigma`, clipped
            to within [lower, upper]. If False, samples are drawn from
            the full prior range.

        Returns
        -------
        float or np.ndarray
            Sampled values from the power-law prior distribution.
        """
        lo, up = self._bounds(initial)

        if n == -1:
            # Special case: P(x) ~ 1/x
            return lo * (up / lo) ** np.random.rand(n)

        # Exponent plus one (aka `pone`)
        pone = self.exponent + 1

        return (
            (np.random.rand(n) * (up ** pone - lo ** pone)) + lo ** pone
        ) ** (1.0 / pone)

    @override
    def evaluate(self, x: float) -> float:
        """
        Evaluates the prior at the sampled value ``x``.

        Parameters
        ----------
        x : float
            The sampled value measured in radians.

        Returns
        -------
        float

        """
        if not self.encompasses(x):
            return -np.inf

        # Exponent plus one (aka `pone`)
        pone = self.exponent + 1

        return (
            pone * x ** self.exponent /
            (self.lower ** pone - self.upper ** pone)
        )

class SinePrior(UniformPrior):
    """
    Sine prior.

    GRBs are statistically much more likely to be
    pointed away from us than toward us. The
    probability of a jet being oriented at an angle
    :math:`\Theta` depends on the solid angle distribution,
    which scales with the area of a spherical cap
    :math:`2\pi sin(\Theta) d\Theta`.

    So that the probability density function for theta
    should be proportional to :math:`sin(\Theta)`.

    Attributes
    ----------
    initial_guess : float
        The expected position in the prior.

    initial_sigma : float
        The expected one-sided sigma of the initial position.
    """
    type = Prior.SINE

    def __init__(
        self,
        lower: float,
        upper: float,
        initial_guess: float = None,
        initial_sigma: float = None
    ):
        super().__init__(lower, upper, initial_guess, initial_sigma)

    @override
    def draw(self, n: int, initial: bool = True) -> float | np.ndarray:
        """"""
        lower, upper = self._bounds(initial)

        cos_min = np.cos(lower)
        cos_max = np.cos(upper)
        cos_theta = cos_min - np.random.rand(n) * (cos_min - cos_max)
        return np.arccos(cos_theta)

    @override
    def evaluate(self, x: float) -> float:
        """
        Evaluates the prior at the sampled value ``x``.

        Parameters
        ----------
        x : float
            The sampled value measured in radians.

        Returns
        -------
        float
            The sine of ``x`` if within bounds else ``-np.inf``.
        """
        if not self.encompasses(x):
            return -np.inf

        return np.sin(x) / (np.cos(self.lower) - np.cos(self.upper))
