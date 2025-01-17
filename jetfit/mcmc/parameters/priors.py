import numpy as np
from scipy import stats
from typing_extensions import override

from jetfit.core.defns.enums import Prior
from jetfit.core.defns.mixins import BoundedMixin
from jetfit.core.utils import paths


def prior_factory(d: dict):
    """

    Parameters
    ----------
    d : dict
        Contains the required key : value pairs for the prior specified
        using the `type` key.

    Returns
    -------
    GaussianPrior or UniformPrior
    """
    prior = Prior(d.get('type'))

    match prior:
        case Prior.GAUSSIAN:
            return GaussianPrior.from_dict(d)

        case Prior.TGAUSSIAN:
            return TruncatedGaussianPrior.from_dict(d)

        case Prior.UNIFORM:
            return UniformPrior.from_dict(d)

        case Prior.SINE:
            return SinePrior.from_dict(d)


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
        if not paths.is_expected_type(mu := d.get('mu'), float):
            raise TypeError('Gaussian mu must be of type float.')

        if not paths.is_expected_type(sigma := d.get('sigma'), float):
            raise TypeError('Gaussian sigma must be of type float.')

        return cls(mu, sigma)

    def draw(self, n: int) -> float | np.ndarray:
        """
        Draws ``n`` samples from the Gaussian distribution.

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
        Evaluates the prior at the sampled value ``x``.

        Parameters
        ----------
        x : float or array_like
            The sampled value.

        Returns
        -------
        float
            The prior evaluated at ``x``.

        Examples
        --------
        Evaluate the Gaussian prior at ``x``:

        >>> prior = GaussianPrior(mu=0.3, sigma=0.1)
        >>> p = prior.evaluate(x)
        """
        return stats.norm.pdf(x, loc=self.mu, scale=self.sigma)


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
        if not paths.is_expected_type(mu := d.get('mu'), (int, float)):
            raise TypeError('TGaussian mu must be of type float.')

        if not paths.is_expected_type(sigma := d.get('sigma'), (int, float)):
            raise TypeError('TGaussian sigma must be of type float.')

        if not paths.is_expected_type(lower := d.get('lower'), (int, float)):
            raise TypeError('TGaussian lower must be of type float.')

        if not paths.is_expected_type(upper := d.get('upper'), (int, float)):
            raise TypeError('TGaussian upper must be of type float.')

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
        class_name = self.__class__.__name__
        return f"{class_name}(lower={self.lower}, upper={self.upper}, initial={self.initial_guess}+/-{self.initial_sigma})"

    @classmethod
    def from_dict(cls, d: dict):
        """
        Creates instance from dict ensuring values are OK.

        Parameters
        ----------
        d : dict
            Includes lower and upper bound information.

        Returns
        -------
        UniformPrior
            Instantiated from dictionary.
        """
        if not paths.is_expected_type(lower := d.get('lower'), (int, float)):
            raise TypeError('Uniform lower must be of type float.')

        if not paths.is_expected_type(upper := d.get('upper'), (int, float)):
            raise TypeError('Uniform upper must be of type float.')

        if not paths.is_expected_type(initial := d.get('initial_guess', None), (int, float), True):
            raise TypeError('Initial guess must be of type float.')

        if not paths.is_expected_type(sigma := d.get('initial_sigma', None), (int, float), True):
            raise TypeError('Initial sigma must be of type float.')

        return cls(lower, upper, initial, sigma)

    def draw(self, n: int, initial: bool = True) -> float | np.ndarray:
        """
        Draws ``n`` samples from the uniform distribution.

        Draws from ``initial_guess`` +/- ``initial_sigma`` if they are both
        defined. Else, draws between ``lower`` and ``upper``.

        Parameters
        ----------
        n : float
            The number of samples to draw.

        initial : bool
            If ``True`` only samples from the initial region (if defined).
            Else, draws from between ``lower`` and ``upper``.

        Returns
        -------
        np.ndarray or float
            Drawn sample(s) from the uniform distribution.
        """
        if initial:
            if self.initial_guess is not None and self.initial_sigma is not None:
                return np.random.uniform(
                    max(self.initial_guess - self.initial_sigma, self.lower),
                    min(self.initial_guess + self.initial_sigma, self.upper),
                    size=n
                )
        return np.random.uniform(self.lower, self.upper, size=n)

    def evaluate(self, x: float) -> float:
        """
        Evaluates the uniform prior at the sampled value ``x``.

        Parameters
        ----------
        x : float
            The sampled value.

        Returns
        -------
        float
            `Zero` if the ``x`` is within the bounds else `-np.inf`.
        """
        return 0.0 if self.encompasses(x) else -np.inf


class SinePrior(UniformPrior):
    """
    Sine prior.

    GRBs are statistically much more likely to be pointed away from us than
    toward us. The probability of a jet being oriented at an angle
    :math:`\Theta` depends on the solid angle distribution, which scales with
    the area of a spherical cap :math:`2\pi sin(\Theta) d\Theta`.

    So that the probability density function for theta should be proportional
    to :math:`sin(\Theta)`.

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
            The sine of ``x`` if it is within the bounds else `-np.inf`.
        """
        return np.sin(x) if self.encompasses(x) else -np.inf
