import emcee
import numpy as np

from jetfit.core.defns.enums import ScaleType
from jetfit.core.defns.observation import Observation
from jetfit.model.parameters import ModelParameters
from jetfit.mcmc.mcmc import MCMC
from jetfit.model import generator
from jetfit.scalers.scaler import Scaler
from jetfit.core.utilities import maths


class JetFit:
    """

    """
    def __init__(self, mcmc: MCMC, observation: Observation, scaler: Scaler):
        self.mcmc = mcmc
        self.scaler = scaler
        self.observation = observation
        self.sampler = self.create_sampler()

    def get_model_parameters(self, theta: np.ndarray[float],
                             scale: ScaleType)-> ModelParameters:
        """
        Returns the JetFit model parameters in `scale` scale.

        Parameters
        ----------
        theta : np.ndarray of float
            The MCMC sampled parameters.

        scale : `jetfit.core.defns.enum.ScaleType`
            The scale to return the parameters in.

        Returns
        -------
        ModelParameters
            The JetFit model parameters in `scale` scale.
        """
        params = {}

        for i, p in enumerate(self.mcmc.fitting_parameters):
            params[p.name] = maths.to_scale(theta[i], p.scale, scale)

        for i, p in enumerate(self.mcmc.fixed_parameters):
            params[p.name] = maths.to_scale(p.value, p.scale, scale)

        return ModelParameters(**params)

    def create_sampler(self) -> emcee.EnsembleSampler:
        """
        Creates and returns an MCMC sampler using `self.mcmc` values.

        Returns
        -------
        `emcee.EnsembleSampler`
            The sampler using `self.mcmc` values.
        """
        return emcee.EnsembleSampler(
            nwalkers=self.mcmc.num_walkers,
            ndim=self.mcmc.num_dims,
            log_prob_fn=self.log_posterior
        )

    def run(self) -> None:
        """
        Runs the MCMC routine.
        """
        self.mcmc.start_run_pos = (
            self.sampler.run_mcmc(
                self.mcmc.start_burn_pos, self.mcmc.burn_length, progress=True)
        )
        self.sampler.reset()

        self.sampler.run_mcmc(
            self.mcmc.start_run_pos, self.mcmc.run_length, progress=True
        )

    def chi_squared(self, params: ModelParameters) -> float:
        """
        Calculates and returns the chi-squared.

        Returns
        -------
        float
            The chi-squared value.
        """
        peak_fluxes, cooling_frequencies, synchrotron_frequencies = (
            self.scaler.scaled_characteristics(params)
        )

        if np.isnan(peak_fluxes[0]):
            return -np.inf

        if peak_fluxes[0] == 0:
            return -np.inf

        chi_squared = 0
        for i, observed in enumerate(self.observation.fluxes):

            modeled = generator.generate(
                observed,
                peak_fluxes[i],
                cooling_frequencies[i],
                synchrotron_frequencies[i],
                params.electron_index
            )

            if np.isnan(modeled.value):
                return -np.inf

            chi_squared += ((observed - modeled).value / observed.avg_error) ** 2

        return chi_squared

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

        for i, p in enumerate(self.mcmc.fitting_parameters):
            if np.isinf(prior := p.prior.evaluate(theta[i])):
                return -np.inf

            log_prior += np.log(prior)

        return log_prior

    def log_likelihood(self, theta: np.ndarray[float]) -> float:
        """
        Calculates the natural log of the likelihood.

        Parameters
        ----------
        theta : np.ndarray of float, with length of `self.mcmc.parameters`
            The sampled MCMC parameter values.

        Returns
        -------
        float
            The log of the likelihood.
        """
        return -0.5 * self.chi_squared(
            self.get_model_parameters(theta, ScaleType.LINEAR)
        )

    def log_posterior(self, theta: np.ndarray[float]) -> float:
        """
        Calculates the natural log of the posterior probability.

        The posterior probability is the probability of the parameters,
        `theta`, given the evidence X denoted by p(theta | X).

        Parameters
        ----------
        theta : np.ndarray of float, with length of `self.mcmc.parameters`
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
