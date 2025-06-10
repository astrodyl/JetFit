import copy
import time

import emcee
import ptemcee
import numpy as np

from jetfit.core.utils import math_utils

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading


class PTSampler:
    """
    Provides an API adapter that matches ``emcee``.

    Parallel tempering in ``emcee`` stopped receiving
    support and was removed from official releases.

    Using the latest version of ``emcee`` that supports
    the PTSampler forces users to use very old packages
    that the PTSampler depends on.

    There is a community developed version called ``ptemcee``.
    However, the authors stopped maintaining it years ago.
    Sadly, there's also zero documentation and/or tutorials,
    and the API is completely different.

    This class aims to provide an API that matches ``emcee``.
    Methods are only added on an as-needed basis and are by
    no means complete.

    Parameters
    ----------
    ntemps : int
        The number of temperatures.

    nwalkers : int
        The number of walkers.

    ndim : int
        The number of fitting dimensions.

    log_like, log_prior
        The log likelihood and log prior methods.

    log_l_args, log_p_args : array_like, optional
        The log likelihood and log prior arguments.

    log_l_kwargs, log_p_kwargs : array_like, optional
        The log likelihood and log prior kwargs.
    """
    def __init__(
        self, ntemps, nwalkers, ndim, log_like, log_prior,
        log_l_args=(), log_p_args=(), log_l_kwargs=(), log_p_kwargs=()
    ):
        # Initialize the sampler
        self._sampler = ptemcee.Sampler(
            nwalkers, ndim, log_like, log_prior,
            log_l_args, log_p_args, log_l_kwargs, log_p_kwargs,
            ptemcee.make_ladder(ndim, ntemps)
        )
        self._chain = None
        self._iteration = 0

        self._ndim = ndim
        self._ntemps = ntemps
        self._nwalkers = nwalkers

    @property
    def sampler(self):
        return self._sampler

    @property
    def chain(self):
        return self._chain

    @property
    def iteration(self):
        return self._iteration

    @property
    def ntemps(self):
        return self._ntemps

    @property
    def nwalkers(self):
        return self._nwalkers

    @property
    def ndim(self):
        return self._ndim

    def run_mcmc(self, x0, iterations, **kwargs):
        """
        Perform MCMC sampling.

        Parameters
        ----------
        x0 : np.ndarray
            The initial position vector.

        iterations : int
            The number of steps to run.

        kwargs
            thin_by

            random

        Returns
        -------
        np.ndarray with shape [ntemps, nwalkers, ndim]
            The last samples.
        """
        self._chain = self.sampler.chain(x0)
        self._chain.run(iterations)
        self._iteration = self._chain.length

        return self.chain.x[-1]

    def reset(self):
        """
        Overwrites the sampler with a new one.

        There's no reset method in ``ptemcee`` that I'm aware
        of. Overwriting with a new sampler is safer than
        attempting to reset attributes individually.
        """
        self._sampler = ptemcee.Sampler(
            self.sampler.nwalkers, self.sampler.ndim,
            self.sampler.logl, self.sampler.logp,
            self.sampler.logl_args, self.sampler.logp_args,
            self.sampler.logl_kwargs, self.sampler.logp_kwargs,
            ptemcee.make_ladder(self.ndim, self.ntemps)
        )
        self._chain = None
        self._iteration = 0

    def get_last_sample(self):
        """ Returns last samples with shape [ntemps, nwalkers, ndim]. """
        if self.chain is None:
            raise AttributeError(
                'Tried to get the last sample, but '
                'there are no samples. Have you '
                'called `run_mcmc` yet?'
            )
        return self.chain.x[-1]

    def get_value(self, name, flat=False, thin=1, discard=0, temp=0):
        """
        Get the attribute ``name``.

        Parameters
        ----------
        name : str
            Name of the attribute to retrieve.

        flat : bool, optional, default=False
            Flatten the chain across the ensemble.

        thin : int, optional, default=1
            Take only every ``thin`` steps from the
            chain.

        discard : int, optional, default=0
            Discard the first ``discard`` steps in the
            chain as burn-in.

        temp : int, optional, default=0
            Take only the ``temp`` attribute. Defaults to
            the temp at index ``0`` which is the highest
            probability temperature.

        Returns
        -------
        np.ndarray
        """
        if self.chain is None:
            raise AttributeError(
                f'Tried to get {name}, but there '
                f'are no chains. Have you called '
                f'`run_mcmc` yet?'
            )

        try:
            v = getattr(self, name)
        except AttributeError:
            v = getattr(self.chain, name)

        if len(v.shape) == 4:
            # shape(iterations, ntemps, nwalkers, ndim)
            v = v[:, temp, :, :]
        else:
            # shape(iterations, ntemps, nwalkers)
            v = v[:, temp, :]

        # Discard and thin
        v = v[discard + thin - 1: self.iteration: thin]

        if flat:
            s = list(v.shape[1:])
            s[0] = np.prod(v.shape[:2])
            return v.reshape(s)
        return v

    def get_chain(self, **kwargs):
        """
        Get the stored chain of MCMC samples.

        Parameters
        ----------
        kwargs
            flat : bool, optional, default=False
                Flatten the chain across the ensemble.

            thin : int, optional, default=1
                Take only every ``thin`` steps from the
                chain.

            discard : int, optional, default=0
                Discard the first ``discard`` steps in the
                chain as burn-in.

            temp : int, optional, default=0
                Take only the ``temp`` chain. Defaults to
                the temp at the ``0``th index which corresponds
                to the highest probability temperature.

        Returns
        -------
        np.ndarray with shape [..., nwalkers, ndim]
            The samples contained in ``ptemcee.Chain.x``.
        """
        return self.get_value('x', **kwargs)

    def get_log_prob(self, **kwargs):
        """
        Get the chain of log probabilities evaluated at
        the MCMC samples.

        Parameters
        ----------
        kwargs
            flat : bool, optional, default=False
                Flatten the chain across the ensemble.

            thin : int, optional, default=1
                Take only every ``thin`` steps from the
                chain.

            discard : int, optional, default=0
                Discard the first ``discard`` steps in the
                chain as burn-in.

            temp : int, optional, default=0
                Take only the ``temp`` log prob. Defaults to
                the temp at the ``0``th index which corresponds
                to the highest probability temperature.

        Returns
        -------
        np.ndarray with shape [..., nwalkers]
            The chain of log probabilities.
        """
        return self.get_value("logP", **kwargs)


class MCMC:
    """

    Parameters
    ----------
    sampler : str, {'emcee', 'ptemcee'}
        The name of the MCMC sampler to use.

    sampler_args : dict
        The required args for the sampler.

    model : ObservedFluxModel
        The afterglow model.

    observation : Observation
        The observational data.

    parameters : Parameters
        The model parameters.

    model_kw : dict, optional
        The optional args for the model.

    sampler_kw : dict, optional
        The optional args for the sampler.
    """
    def __init__(
            self,
            sampler,
            sampler_args,
            model,
            observation,
            parameters,
            model_kw=None,
            sampler_kw=None,
    ):
        # Model
        self.model = model
        self.model_kw = model_kw or {}
        self.params = parameters
        self.observation = observation

        # Sampler
        self.sampler = None
        self.burn_sampler = None

        self.param_pos = None
        self.start_burn_pos = None
        self.start_run_pos = None

        # Setters
        self.set_sampler(sampler, sampler_args, sampler_kw)
        self.set_start_positions(**sampler_args)
        self.set_param_position()

    # <editor-fold desc="Getters and Setters">
    # def start_position(self, nwalkers, ndim, ntemps=None):
    #     """"""
    #     # PTSampler shape == (ntemps, nwalkers, ndim)
    #     if isinstance(self.sampler, PTSampler):
    #         if ntemps is None or ntemps <= 0:
    #             raise ValueError('ntemps must be positive')
    #         pos = np.zeros((ntemps, nwalkers, ndim))
    #
    #         for i in range(ntemps):
    #             for j, p in enumerate(self.params.fitting):
    #                 pos[i, :, j] = p.prior.draw(nwalkers)
    #
    #     # Ensemble shape == (ntemps, nwalkers, ndim)
    #     else:
    #         pos = np.zeros((nwalkers, ndim))
    #
    #         for i, p in enumerate(self.params.fitting):
    #             pos[:, i] = p.prior.draw(nwalkers)
    #
    #     return pos

    def set_param_position(self) -> None:
        """ Sets the position of the parameters in the fitting list. """
        self.param_pos = {p.name : i for i, p in enumerate(self.params.fitting)}

    def set_start_positions(self, nwalkers, ndim, ntemps=None):
        """ Sets the starting burn-in positions. """
        # PTSampler shape == (ntemps, nwalkers, ndim)
        if isinstance(self.sampler, PTSampler):
            if ntemps is None or ntemps <= 0:
                raise ValueError('ntemps must be positive')
            self.start_burn_pos = np.zeros((ntemps, nwalkers, ndim))

            for i in range(ntemps):
                for j, p in enumerate(self.params.fitting):
                    self.start_burn_pos[i, :, j] = p.prior.draw(nwalkers)

        # Ensemble shape == (ntemps, nwalkers, ndim)
        else:
            self.start_burn_pos = np.zeros((nwalkers, ndim))

            for i, p in enumerate(self.params.fitting):
                self.start_burn_pos[:, i] = p.prior.draw(nwalkers)

    def set_sampler(self, sampler, args, kw=None) -> None:
        """
        Initializes the MCMC sampler.

        Parameters
        ----------
        sampler : str
            The name of the MCMC sampler.

        args : dict
            The required args for the sampler.

        kw : dict, optional, default=None
            The optional args for the sampler.
        """
        if sampler not in ('emcee', 'ptemcee'):
            raise ValueError(
                f'Unexpected sampler name: {sampler}. '
                f'Must be either `emcee` or `ptemcee`.'
            )
        kw = kw or {}

        if sampler == 'emcee':
            self.sampler = emcee.EnsembleSampler(
                **args, **kw, log_prob_fn=self.log_posterior)

        else:
            self.sampler = PTSampler(
                **args, **kw, log_like=self.log_likelihood, log_prior=self.log_prior
            )

    def get_best_params(self, as_dict=True, **kwargs):
        """
        Returns the sampled values from the chain with the
        highest likelihood.

        Parameters
        ----------
        as_dict : bool, optional, default=True
            If True, return the sampled values as a dict.

        kwargs : dict
            cat : str, optional
                Limit the params to the ``cat`` categories.

            group : str, optional
                Limit the params to the ``group`` data groups.

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
    def run(self, iterations, burn=0, **kwargs) -> None:
        """
        Performs a burn-in and runs the MCMC routine.

        Parameters
        ----------
        iterations : int
            The number of iterations to run.

        burn : int, optional, default=0
            The length of the burn in.

        kwargs
        """
        # !! TEMP !!
        # If the start position evaluates to -inf, then
        # the samplers throw exception.
        # if isinstance(self.sampler, PTSampler):
        #
        #     log_p = np.full((self.sampler.ntemps, self.sampler.nwalkers), -np.inf)
        #
        #     for i in range(self.sampler.ntemps):
        #         for j in range(self.sampler.nwalkers):
        #             log_p[i, j] = self.log_posterior(self.start_burn_pos[i, j])
        #
        #         best = np.nanargmax(log_p[i])
        #         self.start_burn_pos[i][np.isinf(log_p[i])] = np.array(self.start_burn_pos[i][best], copy=True)

        if burn > 0:
            print('starting single-threaded burn-in')
            start = time.time()
            # Run the burn in and save the last position as the
            # starting position for the actual run.
            self.start_run_pos = (
                self.sampler.run_mcmc(self.start_burn_pos, burn, **kwargs)
            )
            end = time.time()

            print('Burn-in time:', end - start)
            # Save the sampler if desired for diagnostics
            self.burn_sampler = copy.deepcopy(self.sampler)
            self.sampler.reset()

        # Run the sampler
        start = time.time()
        self.sampler.run_mcmc(
            self.start_run_pos, iterations, **kwargs
        )
        end = time.time()
        print('Run time:', end - start)

    def run_mt(self, iterations, burn=0, **kwargs):
        """"""
        with ProcessPoolExecutor(max_workers=10) as pool:
            self.sampler = emcee.EnsembleSampler(
                30, len(self.params.fitting), log_prob_fn=self.log_posterior, pool=pool)

            if burn > 0:
                print('starting multi-threaded burn-in')
                start = time.time()
                # Run the burn in and save the last position as the
                # starting position for the actual run.
                self.start_run_pos = (
                    self.sampler.run_mcmc(self.start_burn_pos, burn, **kwargs)
                )
                end = time.time()

                print('Burn-in time:', end - start)
                # Save the sampler if desired for diagnostics
                self.burn_sampler = copy.deepcopy(self.sampler)
                self.sampler.reset()

            # Run the sampler
            start = time.time()
            self.sampler.run_mcmc(
                self.start_run_pos, iterations, **kwargs
            )
            end = time.time()
            print('Run time:', end - start)

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
        # t0 = time.time()
        # while time.time() - t0 < 0.1:
        #     np.linalg.svd(np.random.rand(100, 100))
        # return -0.5 * np.sum(theta ** 2)

        if np.isfinite(log_prior := self.log_prior(theta)):
            log_likelihood = self.log_likelihood(theta)

            if np.isfinite(log_likelihood):
                return log_prior + log_likelihood

        return -np.inf

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

        # Model the observed afterglow flux
        print(f"[{threading.get_ident()}] Starting JetSimPy at {time.time()}")
        modeled = self.model(self.observation, params, **self.model_kw)
        print(f"[{threading.get_ident()}] Done JetSimPy at {time.time()}")
        # A nan will always result in -inf likelihood, so do a quick
        # check here to avoid unnecessary calculations.
        if np.isnan(modeled.min()):
            return -np.inf

        # Apply calibration offsets
        modeled = self.calibration_offsets(
            modeled, self.params.get(params, 'offsets')
        )

        # return log likelihood
        return -0.5 * self.chi_squared(modeled, self.slop(params))  # type: ignore

    def calibration_offsets(self, modeled, offsets) -> np.ndarray:
        """
        Applies calibration offsets to the modeled values.

        Parameters
        ----------
        modeled : np.ndarray of float
            The modeled values.

        offsets : dict
            Key value pairs of `CalGroup` and offset values [mag].

        Returns
        -------
        np.ndarray
            The modeled values with applied offsets.
        """
        if offsets is not None:
            cal_pos = self.observation.offsets

            for name, offset in offsets.items():
                modeled[cal_pos[name]] *= 10.0 ** -(0.4 * offset)

        return modeled

    def slop(self, params) -> float | np.ndarray | None:
        """
        Formats the slop according to data groups.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        float or np.ndarray of float or None
            The slop value(s).
        """

        # No data groups
        if params.get('shared') is None:
            return params.get('slop').get('slop')

        # Multiple data groups, but only one slop
        if params.get('shared').get('slop'):
            return params.get('shared').get('slop').get('slop')

        # Multiple slops
        s = np.empty(self.observation.length)

        for group, pos in self.observation.groups.items():
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

        # Chi-squared for flux (uses slop)
        flux_mask = self.observation.flux_loc

        cs_flux = math_utils.chi_squared(
            modeled[flux_mask],
            self.observation.as_arrays.values[flux_mask],
            self.observation.as_arrays.errors[flux_mask],
            slop if isinstance(slop, float) else slop[flux_mask]
        )

        # Chi-squared for spectral indices (does not use slop)
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







class MCMC2:
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
            backend=None,
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

        self.backend = backend

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
        """ Sets the starting burn-in positions by drawing from the priors. """
        self.start_burn_pos = np.zeros((self.num_walkers, self.num_dims))

        for i, p in enumerate(self.params.fitting):
            self.start_burn_pos[:, i] = p.prior.draw(self.num_walkers)

    def set_sampler(self, backend=None) -> None:
        """ Initializes the MCMC sampler. """
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.num_walkers,
            ndim=self.num_dims,
            log_prob_fn=self.log_posterior,
            backend=backend
        )

    def get_best_params(self, as_dict=True, **kwargs):
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

    def run_pt(self):
        """"""
        p0 = np.tile(self.start_burn_pos, (10, 1, 1))
        # p0 += 1e-4 * np.random.randn(10, self.num_walkers, self.num_dims)

        chain = self.sampler.chain(p0)
        chain.run(5)

    # <editor-fold desc="Sampling Routine">
    def run(self) -> None:
        """ Performs a burn-in and runs the MCMC routine. """

        # Some walkers can get stuck in a low-probability region
        # To resolve this, Dan Foreman-Mackey recommends:
        #   (1) Run a short chain
        #   (2) Reinitialize the walkers near max log prob
        #   (3) Repeat (1) a few times
        #   (4) Continue on with the actual run
        #
        # See: https://groups.google.com/g/emcee-users/c/fg7sQNw8YcU
        # start_pos = np.array(self.start_burn_pos, copy=True)
        #
        # for i in range(5):
        #     self.sampler.run_mcmc(start_pos, 200, progress=True)
        #     max_index = np.nanargmax(self.sampler.get_log_prob())
        #     max_log_prob = self.sampler.get_chain(flat=True)[max_index]
        #
        #     # Center the walkers on the highest likelihood region found
        #     # so far using a gaussian distribution.
        #     for j, p in enumerate(self.params.fitting):
        #         start_pos[:, j] = GaussianPrior(
        #             mu=max_log_prob[j], sigma=0.05 * (p.prior.upper - p.prior.lower)
        #         ).draw(self.num_walkers)
        #
        #     self.sampler.reset()

        # Run the burn in and save the last position as the
        # starting position for the actual run.
        self.start_run_pos = (
            self.sampler.run_mcmc(
                self.start_burn_pos, self.burn_length, progress=True,
            )
        )

        # Save the sampler if desired for diagnostics
        self.burn_sampler = copy.deepcopy(self.sampler)
        self.sampler.reset()

        # Update backend
        if self.backend is not None:
            self.backend.reset(self.num_walkers, self.num_dims)
        self.set_sampler(self.backend)

        # Run the sampler
        self.sampler.run_mcmc(
            self.start_run_pos, self.run_length, progress=True
        )

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

        # Model the observed afterglow flux
        modeled = self.model(self.observation, params, **self.meta)

        # A nan will always result in -inf likelihood, so do a quick
        # check here to avoid unnecessary calculations.
        if np.isnan(modeled.min()):
            return -np.inf

        # Apply calibration offsets
        modeled = self.calibration_offsets(
            modeled, self.params.get(params, 'offsets')
        )

        # return log likelihood
        return -0.5 * self.chi_squared(modeled, self.slop(params))  # type: ignore

    def calibration_offsets(self, modeled, offsets) -> np.ndarray:
        """
        Applies calibration offsets to the modeled values.

        Parameters
        ----------
        modeled : np.ndarray of float
            The modeled values.

        offsets : dict
            Key value pairs of `CalGroup` and offset values [mag].

        Returns
        -------
        np.ndarray
            The modeled values with applied offsets.
        """
        if offsets is not None:
            cal_pos = self.observation.offsets

            for name, offset in offsets.items():
                modeled[cal_pos[name]] *= 10.0 ** -(0.4 * offset)

        return modeled

    def slop(self, params) -> float | np.ndarray | None:
        """
        Formats the slop according to data groups.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        float or np.ndarray of float or None
            The slop value(s).
        """

        # No data groups
        if params.get('shared') is None:
            return params.get('slop').get('slop')

        # Multiple data groups, but only one slop
        if params.get('shared').get('slop'):
            return params.get('shared').get('slop').get('slop')

        # Multiple slops
        s = np.empty(self.observation.length)

        for group, pos in self.observation.groups.items():
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

        # Chi-squared for flux (uses slop)
        flux_mask = self.observation.flux_loc

        cs_flux = math_utils.chi_squared(
            modeled[flux_mask],
            self.observation.as_arrays.values[flux_mask],
            self.observation.as_arrays.errors[flux_mask],
            slop if isinstance(slop, float) else slop[flux_mask]
        )

        # Chi-squared for spectral indices (does not use slop)
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
