import copy
import os

import emcee
import numpy as np

from jetfit.core.utils import math_utils


class MCMC:
    """

    Attributes
    ----------
    burn_length : int
        Number of iterations for burn in.

    run_length : int
        Number of iterations for run after burn in.

    num_walkers : int
        Number of MCMC walkers.

    model : ObservedFluxModel
        The model to evaluate.

    observation : Observation
        The observational data.
    """
    def __init__(
            self,
            burn_length: int,
            run_length: int,
            num_walkers: int,
            model,
            observation,
            parameters,
            filename=None,
            meta=None,
    ):
        # Model
        self.model = model
        self.params = parameters

        # Sampler
        self.sampler = None
        self.burn_sampler = None
        self.run_length = run_length
        self.burn_length = burn_length
        self.num_walkers = num_walkers

        self.param_pos = None
        self.start_burn_pos = None
        self.start_run_pos = None

        # Evidence
        self.observation = observation

        self.filename = filename

        # Setters
        self.set_sampler()
        self.set_start_positions()
        self.set_param_position()

        self.meta = meta if meta else {}

    @property
    def num_dims(self) -> int | None:
        """ The number of fitting parameters. """
        if self.params.fitting is not None:
            return len(self.params.fitting)

    # <editor-fold desc="Getters and Setters">
    def set_param_position(self) -> None:
        """ Sets the position of the parameters in the fitting list. """
        self.param_pos = {p.name : i for i, p in enumerate(self.params.fitting)}

    def set_start_positions(self) -> None:
        """ Calculates the start positions for each MCMC walker. """
        self.start_burn_pos = np.zeros((self.num_walkers, self.num_dims))

        for i, p in enumerate(self.params.fitting):
            self.start_burn_pos[:, i] = p.prior.draw(self.num_walkers)

    def set_sampler(self) -> None:
        """ Initializes the MCMC sampler. """
        backend = None

        if self.filename is not None:
            backend = emcee.backends.HDFBackend(self.filename)
            if os.path.exists(self.filename):
                backend.reset(self.num_walkers, self.num_dims)

        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.num_walkers,
            ndim=self.num_dims,
            log_prob_fn=self.log_posterior,
            # moves=emcee.moves.DEMove(),
            backend=backend  # type: ignore
        )

    def get_best_params(self, as_dict=True,**kwargs):
        """
        Returns the sampled values from the chain with the highest
        likelihood.

        Parameters
        ----------
        as_dict : bool, optional, default=True
            If True, return the sampled values as a dictionary.

        kwargs : dict
            cat : str, optional
                Limit the params to the `cat` categories.

            group : str, optional
                Limit the params to the `group` data groups.

            scale : str, optional, default='linear'
                The scale to return the parameters in.

        Returns
        -------
        dict or np.ndarray
            The values from the highest likelihood chain.
        """
        max_index = np.nanargmax(self.sampler.get_log_prob(flat=True))
        params = self.sampler.get_chain(flat=True)[max_index]
        return self.params.samples_to_dict(params, **kwargs) if as_dict else params
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

        self.burn_sampler = copy.deepcopy(self.sampler)
        self.sampler.reset()

        self.sampler.run_mcmc(
            self.start_run_pos,
            self.run_length,
            progress=True
        )

    def log_prior(self, theta: np.ndarray[float]) -> float:
        """
        Evaluates the natural log of the priors.

        Parameters
        ----------
        theta : np.ndarray of float, with length of `fitting_params`
            The sampled MCMC parameter values.

        Returns
        -------
        float
            The log of the evaluated priors.
        """
        log_prior = 0

        for i, p in enumerate(self.params.fitting):
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
        theta : np.ndarray of float, with length of `fitting_params`
            The sampled MCMC parameter values.

        Returns
        -------
        float or -np.inf
            The log of the likelihood if the parameters were valid.
            Else, -np.inf.
        """
        params = self.params.samples_to_dict(theta)

        modeled = self.model(self.observation, params, **self.meta)

        # Apply calibration offsets
        offsets = params.get('shared').get('offsets') \
            if params.get('shared') is not None else params.get('offsets')

        modeled = self.calibration_offsets(
            modeled, offsets
        )

        # Skip chi squared calculation since a nan will
        # always result in -inf anyway
        if np.isnan(modeled.min()):
            return -np.inf

        # return log likelihood
        return -0.5 * self.chi_squared(modeled, self.slop(params))  # type: ignore

    def log_posterior(self, theta: np.array) -> float:
        """
        Calculates the natural log of the posterior
        probability.

        The posterior probability is the probability
        of the parameters, `theta`, given the evidence
        X denoted by p(theta | X).

        Parameters
        ----------
        theta : np.ndarray of float
            The sampled MCMC parameter values.

        Returns
        -------
        float
            The natural log of the posterior.
        """
        if np.isfinite(log_prior := self.log_prior(theta)):
            log_likelihood = self.log_likelihood(theta)

            if np.isfinite(log_likelihood):
                return log_prior + log_likelihood

        return -np.inf

    def calibration_offsets(self, modeled, offsets) -> np.ndarray:
        """
        Applies calibration offsets to the modeled values.

        Parameters
        ----------
        modeled : np.ndarray
            The modeled values.

        offsets : dict
            Key value pairs of `CalGroup` and offset values.

        Returns
        -------
        np.ndarray
            The modeled values with applied offsets.
        """
        if offsets is not None:
            cal_pos = self.observation.cal_offsets

            for name, offset in offsets.items():
                modeled[cal_pos[name]] *= 10.0 ** -(0.4 * offset)

        return modeled

    def slop(self, params) -> float | np.ndarray:
        """
        Formats the slop according to data groups.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        float or np.ndarray of float
            The slop value(s).
        """

        # No data groups
        if params.get('shared') is None:
            return params.get('slop').get('slop')

        # Multiple data groups, but only one slop
        if params.get('shared').get('slop') is not None:
            return params.get('shared').get('slop').get('slop')

        # Multiple slops
        s = np.empty(self.observation.length)

        for group, pos in self.observation.data_groups.items():
            s[pos] = params.get(group).get('slop').get('slop')

        return s

    def chi_squared(self, modeled, slop=None) -> float:
        """
        Calculates the combined chi-squared between
        the modeled and observational data for both
        the flux and spectral indices.

        The flux chi-squared calculation uses a so-
        called chi-squared effective which utilizes
        a slop parameter. Spectral indices use the
        standard chi-squared formulation.

        Parameters
        ----------
        modeled : np.ndarray of float
            The modeled or predicted values.

        slop : float or np.ndarray of float, optional
            The slop value.

        Returns
        -------
        float
            The combined chi-squared value.
        """

        # Handle flux and indices the same
        if slop is None:
            return math_utils.chi_squared(
                modeled,
                self.observation.as_arrays.values,
                self.observation.as_arrays.errors,
            )

        # Chi-squared for flux
        flux_mask = self.observation.flux_loc

        cs_flux = math_utils.chi_squared(
            modeled[flux_mask],
            self.observation.as_arrays.values[flux_mask],
            self.observation.as_arrays.errors[flux_mask],
            slop if isinstance(slop, float) else slop[flux_mask]
        )

        # Chi-squared for spectral indices
        index_mask = self.observation.sindex_loc

        if not index_mask.any():
            return cs_flux

        cs_indices = math_utils.chi_squared(
            modeled[index_mask],
            self.observation.as_arrays.values[index_mask],
            self.observation.as_arrays.errors[index_mask],
        )

        # return combined chi-squared
        return cs_flux + cs_indices
    # </editor-fold>
