import numpy as np
from scipy import stats
from typing_extensions import override

from jetfit.core.defns.enums import Prior
from jetfit.core.defns.mixins import BoundedMixin
from jetfit.core.utilities import maths, utils


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
        if not utils.is_expected_type(mu := d.get('mu'), float):
            raise TypeError('Gaussian mu must be of type float.')

        if not utils.is_expected_type(sigma := d.get('sigma'), float):
            raise TypeError('Gaussian sigma must be of type float.')

        return cls(mu, sigma)

    def draw(self, n: int):
        """

        Returns
        -------

        """
        return np.random.normal(self.mu, self.sigma, size=n)

    def evaluate(self, x: float) -> float:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float
            The sampled value.

        Returns
        -------
        float
            The prior evaluated at `x`.
        """
        return maths.gaussian(x, self.mu, self.sigma)


class TruncatedGaussianPrior(GaussianPrior, BoundedMixin):
    """
    TruncatedGaussianPrior

    The `TruncatedGaussianPrior` is bounded such that the probability is
    evaluated as a `GaussianPrior` within the bounds and -infinity outside.
    """
    def __init__(self, mu: float, sigma: float, lower: float, upper: float):
        BoundedMixin.__init__(self, lower, upper)
        super().__init__(mu, sigma)

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
        if not utils.is_expected_type(mu := d.get('mu'), float):
            raise TypeError('TGaussian mu must be of type float.')

        if not utils.is_expected_type(sigma := d.get('sigma'), float):
            raise TypeError('TGaussian sigma must be of type float.')

        if not utils.is_expected_type(lower := d.get('lower'), float):
            raise TypeError('TGaussian lower must be of type float.')

        if not utils.is_expected_type(upper := d.get('upper'), float):
            raise TypeError('TGaussian upper must be of type float.')

        return cls(mu, sigma, lower, upper)

    def draw(self, n: int):
        """

        Returns
        -------
        ndarray or scalar
            Drawn samples from the parameterized normal distribution.
        """
        lower = (self.lower - self.mu) / self.sigma
        upper = (self.upper + self.mu) / self.sigma

        return stats.truncnorm.rvs(lower, upper, loc=self.mu, scale=self.sigma, size=n)

    def evaluate(self, x: float) -> float:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float
            The sampled value.

        Returns
        -------
        float
            The prior evaluated at `x` if the sampled values is within the
            `bounds`, else -`np.inf`.
        """
        return super().evaluate(x) if self.encompasses(x) else -np.inf


class UniformPrior(BoundedMixin):
    """
    Uniform prior.
    """
    type = Prior.UNIFORM

    def __init__(self, lower: float, upper: float):
        """

        Parameters
        ----------
        lower : float
            The lower bound.

        upper : float
            The upper bound.
        """
        BoundedMixin.__init__(self, lower, upper)

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
        if not utils.is_expected_type(lower := d.get('lower'), float):
            raise TypeError('Uniform lower must be of type float.')

        if not utils.is_expected_type(upper := d.get('upper'), float):
            raise TypeError('Uniform upper must be of type float.')

        return cls(lower, upper)

    def draw(self, n: int):
        """

        Parameters
        ----------
        n : float
            The number of samples to draw.

        Returns
        -------
        ndarray or float
            Drawn samples from the parameterized uniform distribution.
        """
        return np.random.uniform(self.lower, self.upper, size=n)

    def evaluate(self, x: float) -> float:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float
            The sampled value.

        Returns
        -------
        float
            One if the `x` is within the bounds else `-np.inf`.
        """
        return 1.0 if self.encompasses(x) else -np.inf


class SinePrior(UniformPrior):
    """

    """
    def __init__(self, lower: float, upper: float):
        """

        Parameters
        ----------
        lower : float
            The lower bound.

        upper : float
            The upper bound.
        """
        super().__init__(lower, upper)

    @override
    def evaluate(self, x: float) -> float:
        """
        Evaluates the prior at the sampled value `x`.

        Parameters
        ----------
        x : float
            The sampled value.

        Returns
        -------
        float
            The sine of `x` if it is within the bounds else `-np.inf`.
        """
        return np.sin(x) if self.encompasses(x) else -np.inf
