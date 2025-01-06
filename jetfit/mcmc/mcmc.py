from pathlib import Path

import emcee
import numpy as np

from jetfit.core.defns.evidence import Evidence
from jetfit.core.utils import maths
from jetfit.mcmc.parameters.parameters import MCMCFittingParameter


class MCMC:
    """

    Attributes
    ----------
    burn_length : int
        Number of iterations for each burn in.

    run_length : int
        Number of iterations for run after burn in.

    num_walkers : int
        Number of MCMC walkers.

    fixed_params : list of MCMCFixedParameter

    fitting_params : list of MCMCFittingParameter

    model :
        ??

    evidence : Evidence

    """
    def __init__(
            self,
            burn_length: int,
            run_length: int,
            num_walkers: int,
            evidence: Evidence,
            fixed_params: list,
            fitting_params: list,
            model
    ):
        # Model
        self.model = model
        self.fixed_params = fixed_params
        self.fitting_params = fitting_params
        self.param_pos = None

        # Sampler
        self.sampler = None
        self.run_length = run_length
        self.burn_length = burn_length
        self.num_walkers = num_walkers

        self.start_burn_pos = None
        self.start_run_pos = None

        # Evidence
        self.evidence = evidence

        # Setters
        self.set_sampler()
        self.set_start_positions()
        self.set_param_position()

    @property
    def num_dims(self) -> int | None:
        """ The number of fitting parameters """
        if self.fitting_params is not None:
            return len(self.fitting_params)

    @classmethod
    def from_toml(cls, mcmc_path: str | Path, model_path: str | Path):
        """
        Instantiates the class from a TOML files.

        Parameters
        ----------
        mcmc_path : str or Path
            Path to the MCMC settings TOML file.

        model_path : str or Path
            Path to the model parameters TOML file.

        Returns
        -------
        MCMC
            Instantiated MCMC class from TOML files.
        """
        pass

    # <editor-fold desc="Getters and Setters">
    def set_param_position(self) -> None:
        """ Sets the position of the parameters in the fitting list. """
        self.param_pos = {
            p.name : i for i, p in enumerate(self.fitting_params)
        }

    def set_start_positions(self) -> None:
        """ Calculates the start positions for each MCMC walker. """
        self.start_burn_pos = np.zeros((self.num_walkers, self.num_dims))

        for i, p in enumerate(self.fitting_params):
            self.start_burn_pos[:, i] = p.prior.draw(self.num_walkers)

    def set_sampler(self) -> None:
        """ Initializes the MCMC sampler. """
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.num_walkers,
            ndim=self.num_dims,
            log_prob_fn=self.log_posterior,
        )

    def get_slop(self, theta: np.ndarray[float]) -> None | float:
        """
        Indexes the array of samples and returns the slope value.

        Parameters
        ----------
        theta
            The `emcee.Emcee` sample array.

        Returns
        -------
        float or None
            The slop value if it is a fitting parameter, else None.
        """
        if (pos := self.param_pos.get('slop')) is not None:
            return theta[pos]

    def get_best_params(self, obj: bool = True):
        """
        Returns the sampled values from the chain with the highest likelihood.

        Parameters
        ----------
        obj : bool, optional, default=True
            If ``True``, returns the parameters object associated with the
            ``model`` attribute. Else, returns a numpy array of values.

        Returns
        -------
        ModelParams or np.ndarray of float
            The values from the highest likelihood chain.
        """
        max_index = np.nanargmax(
            self.sampler.get_log_prob(flat=True)
        )
        params = self.sampler.get_chain(flat=True)[max_index]

        if obj:
            return self.model.parameters.from_mcmc_samples(
                self.fixed_params, self.fitting_params, params
            )

        return params
    # </editor-fold>

    # <editor-fold desc="Sampling Routine">
    def run(self) -> None:
        """ Performs a burn-in and runs the MCMC routine. """
        self.start_run_pos = (
            self.sampler.run_mcmc(
                self.start_burn_pos,
                self.burn_length,
                progress=True,
            )
        )
        self.sampler.reset()

        self.sampler.run_mcmc(
            self.start_run_pos,
            self.run_length,
            progress=True
        )

    def log_prior(self, theta: np.ndarray[float]) -> float:
        """
        Calculates the natural log of the likelihood.

        Parameters
        ----------
        theta : np.ndarray of float, with length of `self.mcmc.parameters`
            The sampled MCMC parameter values.

        Returns
        -------
        float
            The log of the evaluated priors.
        """
        log_prior = 0

        for i, p in enumerate(self.fitting_params):
            if np.isinf(prior := p.prior.evaluate(theta[i])):
                return -np.inf

            if prior != 0:
                log_prior += np.log(prior)

        return log_prior

    def log_likelihood(self, theta: np.ndarray[float]) -> float:
        """
        Calculates the natural log of the likelihood.

        Parameters
        ----------
        theta : np.ndarray of float, with length of ``fitting_params``
            The sampled MCMC parameter values.

        Returns
        -------
        float
            The log of the likelihood.
        """
        params = self.model.parameters.from_mcmc_samples(
            self.fixed_params, self.fitting_params, theta
        )

        if np.isnan((modeled := self.model.evaluate(self.evidence, params))[0]):
            # Typically, array is all NaN or all numbers.
            return -np.inf

        return -0.5 * maths.chi_squared(
            modeled,
            self.evidence.optimized_y,
            self.evidence.optimized_err,
            self.get_slop(theta)
        )

    def log_posterior(self, theta: np.ndarray[float]) -> float:
        """
        Calculates the natural log of the posterior probability.

        The posterior probability is the probability of the parameters,
        ``theta``, given the evidence X denoted by p(theta | X).

        Parameters
        ----------
        theta : np.ndarray of float, with length of ``fitting_params``
            The sampled MCMC parameter values.

        Notes
        -----
        This method is called `num_walkers` x `num_iterations` times which is
        typically 1e6 times. Iterating even once over a dataset with 1e3
        datapoints results in 1e9 (1 billion) iterations. Developers must
        respect this expense when modifying this method.

        To reduce unnecessary calculations, I only calculate the likelihood if
        the prior is a finite value since there is no possible value of the
        likelihood that could modify `-infinity`.

        Returns
        -------
        float
            The natural log of the posterior.
        """
        if np.isfinite(log_prior := self.log_prior(theta)):
            if np.isfinite(likelihood := self.log_likelihood(theta)):
                return log_prior + likelihood

        return -np.inf
    # </editor-fold>
