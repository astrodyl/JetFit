import copy
from contextlib import nullcontext

import emcee
import ptemcee
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from jetfit.core.utils import math_utils


def get_pool_context(workers=None, executor='process'):
    """
    If ``executor==process`` and ``workers>1``:
        Returns ``ProcessPoolExecutor(max_workers=workers)``.

        Since processes need to load everything into memory, this
        should only be used if the likelihood calculation takes
        about one-second or more to calculate.

    If ``executor==thread`` and ``workers>1``:
        Returns ``ThreadPoolExecutor(max_workers=workers)``.

        Note that unless a free-threaded Python is installed,
        multithreading will not yield any benefits. Even if a
        no-GIL Python version is used, the performance increase
        depends  on the likelihood implementation.

        Pure Python implementations will see a large performance
        increase. If the likelihood uses Cython, then it depends
        on how the code is compiled and optimized.

    Parameters
    ----------
    workers : int, optional, default=None
        The max number of workers.

    executor : str, optional, default='process'
        See above docstring for details. Must be ``process``
        or ``thread``.

    Returns
    -------
    ``ProcessPoolExecutor`` or ``ThreadPoolExecutor`` or ``nullcontext``
        The pool context manager.
    """
    if workers and workers > 1:

        if executor == 'process':
            return ProcessPoolExecutor(max_workers=workers)

        if executor == 'thread':
            return ThreadPoolExecutor(max_workers=workers)

    return nullcontext()


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
    Performs MCMC sampling.

    Parameters
    ----------
    model : MCMCModels
        The afterglow model.

    observation : Observation
        The observational data.

    parameters : Parameters
        The model parameters.
    """
    def __init__(self, model, observation, parameters):
        # Model
        self.models = model
        self.params = parameters
        self.observation = observation

        # Sampler
        self.sampler = None
        self.burn_chain = None

        self.param_pos = None
        self.start_burn_pos = None
        self.start_run_pos = None

        # Setters
        self.set_param_position()

    # <editor-fold desc="Getters and Setters">
    @property
    def ndim(self):
        """ The number of fitting dimensions. """
        return len(self.params.fitting)

    def set_param_position(self) -> None:
        """ Sets the position of the parameters in the fitting list. """
        self.param_pos = {p.name : i for i, p in enumerate(self.params.fitting)}

    def set_start_positions(self, nwalkers, ntemps=None):
        """
        Sets the starting burn-in positions.

        Parameters
        ----------
        nwalkers : int
            The number of walkers.

        ntemps : int, optional, default=None
            The number of temperatures for ``ptemcee``.
        """
        # PTSampler shape == (ntemps, nwalkers, ndim)
        if isinstance(self.sampler, PTSampler):
            self.start_burn_pos = np.zeros((ntemps, nwalkers, self.ndim))

            for i in range(ntemps):
                for j, p in enumerate(self.params.fitting):
                    self.start_burn_pos[i, :, j] = p.prior.draw(nwalkers)

            # Overwrite invalid posteriors with best for each temp
            log_p = np.full((ntemps, nwalkers), -np.inf)
            for i in range(ntemps):
                for j in range(nwalkers):
                    log_p[i, j] = log_posterior(
                        self.start_burn_pos[i, j], self.params, self.models
                    )

                best = np.nanargmax(log_p[i])
                self.start_burn_pos[i][np.isinf(log_p[i])] = np.array(
                    self.start_burn_pos[i][best], copy=True
                )

        # Ensemble shape == (ntemps, nwalkers, ndim)
        else:
            self.start_burn_pos = np.zeros((nwalkers, self.ndim))

            for i, p in enumerate(self.params.fitting):
                self.start_burn_pos[:, i] = p.prior.draw(nwalkers)

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
    @staticmethod
    def _validate_ptemcee(ntemps, pool=None):
        """"""
        if ntemps is None:
            raise ValueError(
                "Sampler `ptemcee` requires arge `ntemps`."
            )

        if pool is not None:
            raise ValueError(
                "Sampler `ptemcee` does not support"
                "multi-processing/threading."
            )

    @staticmethod
    def _validate_emcee(ntemps=None):
        """"""
        if ntemps is not None:
            raise ValueError(
                "Sampler `emcee` does not use `ntemps`."
            )

    def _validate_sampler(self, sampler, pool=None, ntemps=None):
        """"""
        if sampler not in ('emcee', 'ptemcee'):
            raise ValueError(
                f'Unexpected sampler name: {sampler}. '
                f'Must be either `emcee` or `ptemcee`.'
            )

        if sampler == 'ptemcee':
            self._validate_ptemcee(ntemps, pool)

        if sampler == 'emcee':
            self._validate_emcee(ntemps)

    def set_sampler(self, sampler, nwalkers, pool, ntemps=None, **kwargs):
        """
        Sets the sampler. Duh.

        Parameters
        ----------
        sampler : str
            The sampler type. Must be ``emcee`` or ``ptemcee``.

        nwalkers : int
            The number of walkers.

        pool : ``ProcessPoolExecutor`` or ``ThreadPoolExecutor`` or ``nullcontext``
            The pool to use for multithreading/processing.
            See ``get_pool_context()`` for details.

        ntemps : int, optional
            The number of temperatures for ``ptemcee``.

        kwargs
            Any kwargs to be passed to the sampler.
        """
        self._validate_sampler(sampler, pool, ntemps)

        if sampler == 'emcee':
            if isinstance(pool, type(nullcontext())):
                # Single threaded/processed
                self.sampler = emcee.EnsembleSampler(
                    nwalkers, self.ndim, log_posterior,
                    args=(self.params, self.models), **kwargs  # type: ignore
                )
            else:
                # Multithreaded/processed
                self.sampler = emcee.EnsembleSampler(
                    nwalkers, self.ndim, log_posterior,
                    args=(self.params, self.models), pool=pool, **kwargs # type: ignore
                )

        elif sampler == 'ptemcee':
            # PTSampler is single processed only
            self.sampler = PTSampler(
                ntemps, nwalkers, self.ndim, log_likelihood, log_prior,
                log_l_args=(self.params, self.models), log_p_args=(self.params,)
            )

    def run(
        self, nwalkers, iterations, burn=0, sampler='emcee',
        workers=None, ntemps=None, sampler_kw=None, run_kw=None
    ):
        """
        Runs the MCMC sampling routine.

        Parameters
        ----------
        nwalkers : int
            The number of walkers.

        iterations : int
            The number of iterations.

        burn : int, optional, default=0
            The number of iterations to burn. If ``burn>0``,
            stores the burn sampler to ``self.burn_sampler``
            before resetting it for the main run.

        sampler : str, optional, default='emcee'
            Must be either `emcee` or `ptemcee`.

        workers : int, optional, default=None
            The max number of workers to use.

        ntemps : int, optional, default=None
            The number of temperatures to use for ``ptemcee``.

        sampler_kw : dict, optional
            Any kwargs to pass to the sampler.

        run_kw : dict, optional
            Any kwargs to pass to the ``run_mcmc`` method.
        """
        with get_pool_context(workers) as pool:
            self.set_sampler(sampler, nwalkers, pool, ntemps, **(sampler_kw or {}))
            self.set_start_positions(nwalkers, ntemps)

            if burn > 0:
                # Run the burn in and save the last position
                self.start_run_pos = (
                    self.sampler.run_mcmc(self.start_burn_pos, burn, **(run_kw or {}))
                )

                # Save the chain if desired for diagnostics. Cannot
                # save the entire sampler because deepcopy detaches
                # the pool which prevents multiprocessing/threading
                self.burn_chain = copy.deepcopy(self.sampler.get_chain())
                self.sampler.reset()

            # Run production
            self.sampler.run_mcmc(
                self.start_run_pos, iterations, **(run_kw or {})
            )


class MCMCModels:
    """
    Container for MCMC models used during fitting.

    Parameters
    ----------
    obs : Observation
        The observational data.

    afg_model :
        The afterglow model.

    afg_kw : dict, optional
        Any kwargs needed to instantiate the model.

    ext_model : optional
        The dust extinction model.

    ext_mw_pc : np.ndarray, optional
        The pre-computed Milky Way extinction values.

    ext_sf_pc : np.ndarray, optional
        The pre-computed source-frame extinction values.
    """
    def __init__(
        self, obs, afg_model,
        afg_kw=None, ext_model=None, ext_mw_pc=None, ext_sf_pc=None
    ):
        # Afterglow
        self.afg_model = afg_model
        self.afg_kw = afg_kw if afg_kw else {}

        # Extinction
        self.ext_model = ext_model
        self.ext_mw_pc = ext_mw_pc
        self.ext_sf_pc = ext_sf_pc

        # Observation
        self.obs = obs

    def model(self, params):
        """
        Models the observed GRB afterglow flux.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The modeled observed GRB afterglow flux.
        """
        # Model the GRB afterglow flux
        modeled = self.model_afterglow(params)

        if np.isnan(modeled.min()):
            return np.array([np.nan])

        # Correct for dust and host then return
        return self.model_extinction(modeled, params)

    def model_afterglow(self, params):
        """
        Models the unextinguished GRB afterglow flux.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The modeled GRB afterglow flux.
        """
        if params.get('shared') is not None:
            # Model the data with sets of parameters
            # applied to different subsets of the data.
            modeled = self.model_grouped_afterglow(params)

        else:
            # Model the data all together. Nice and simple.
            model = self.afg_model(**params.get('model'), **self.afg_kw)
            modeled = model.model(self.obs)

        return modeled

    def model_grouped_afterglow(self, params):
        """
        Models the unextinguished GRB afterglow flux divided
        into an arbitrarily defined number of subsets.

        This method is useful for light curves that display
        different behaviors in different temporal regimes.
        Note that it is up to the user to determine whether
        fitting multiple sets of parameters is physically
        meaningful or not. This method simply provides the
        ability to do so.

        Parameters
        ----------
        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The modeled unextinguished GRB afterglow data.
        """
        modeled = np.full(self.obs.length, np.nan, dtype=float)

        for group, mask in self.obs.groups.items():
            model_params = params.get(group).get('model')

            modeled[mask] = self.afg_model(
                **model_params, **self.afg_kw).model(self.obs, mask)

        return modeled

    def model_extinction(self, modeled, params):
        """
        Corrects the afterglow flux, ``modeled``, for
        dust extinction and host galaxy contributions.

        Applies the corrections in the order:
            1. Source-frame dust extinction.
            2. Host galaxy contribution.
            3. Milky Way dust extinction.

        Parameters
        ----------
        modeled : np.array
            The modeled flux.

        params : dict
            The dict returned from `Parameters.samples_to_dict`.

        Returns
        -------
        np.ndarray of float
            The extinguished and host galaxy corrected flux.
        """
        pos = self.obs.extinguishable
        wn = self.obs.as_arrays.wave_numbers[pos]

        if 'shared' in params:
            params = params['shared']

        # Extinction params TEMP!
        z = params.get('model').get('z')
        ext = params.get('extinction')
        ebv_sf = ext.get('ebv_source_frame')
        ebv_mw = ext.get('ebv_milky_way')

        # Apply source-frame extinction
        if ebv_sf is not None:
            p = {'init': {'Rv': ext.get('rv_source_frame') or 3.1}, 'eval': {'Ebv': ebv_sf}}
            modeled[pos] *= self._model_extinction(p, (1 + z) * wn, self.ext_sf_pc)

        # Apply host galaxy correction
        if params.get('host') is not None and self.obs.hosts is not None:
            for name, corr in params.get('host').items():
                modeled[self.obs.hosts[name]] += corr

        # Apply Milky Way extinction
        if ebv_mw is not None:
            p = {'init': {'Rv': ext.get('rv_milky_way') or 3.1}, 'eval': {'Ebv': ebv_mw}}
            modeled[pos] *= self._model_extinction(p, wn, self.ext_mw_pc)

        # return corrected flux.
        return modeled

    def _model_extinction(self, p, wn, pc=None):
        """ Internal use only. """
        # Return the pre-computed extinction
        if pc is not None: return pc

        # Calculate the extinction and return
        return self.ext_model(
            **p.get('init')).extinguish(wn, **p.get('eval'))


# For multiprocessing purposes, emcee requires that methods and
# arguments be pickle-able. As such, the methods below are made
# global to meet this requirement.

def log_prior(theta, params) -> float:
    """
    Evaluates the natural log of the priors.

    Parameters
    ----------
    theta : np.ndarray of float, with length of `fitting_params`
        The sampled MCMC parameter values.

    params : Parameters
        MCMC parameter container.

    Returns
    -------
    float
        The log of the evaluated priors.
    """
    lp = 0

    for i, p in enumerate(params.fitting):
        if np.isinf(prior := p.prior.evaluate(theta[i])):
            return -np.inf

        if prior != 0:
            lp += np.log(prior)

    return lp


def log_likelihood(theta, params, models) -> float:
    """
    Calculates the natural log of the likelihood.

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    params : Parameters
        The MCMC parameter container.

    models : MCMCModels
        The MCMC models container.

    Returns
    -------
    float or -np.inf
        The log of the likelihood if the parameters were valid.
        Else, -np.inf.
    """
    p = params.samples_to_dict(theta)

    # Model the observed afterglow flux
    modeled = models.model(p)

    # A nan will always result in -inf likelihood, so do a quick
    # check here to avoid unnecessary calculations.
    if np.isnan(modeled.min()):
        return -np.inf

    # Apply calibration offsets
    modeled = calibration_offsets(
        modeled, params.get(p, 'offsets'), models.obs.offsets
    )

    # return log likelihood
    return -0.5 * chi_squared(modeled, models.obs, slop(p, models.obs))  # type: ignore


def log_posterior(theta, params, models) -> float:
    """
    Calculates the natural log of the posterior
    probability.

    The posterior probability is the probability
    of the parameters, `theta`, given the evidence
    X denoted by p(theta | X).

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    params : Parameters
        The MCMC parameter container.

    models : MCMCModels
        The MCMC models container.

    Returns
    -------
    float
        The natural log of the posterior.
    """
    if np.isfinite(lp := log_prior(theta, params)):
        ll = log_likelihood(theta, params, models)

        if np.isfinite(ll):
            return lp + ll

    return -np.inf


def calibration_offsets(modeled, offsets, pos) -> np.ndarray:
    """
    Applies calibration offsets to the modeled values.

    Parameters
    ----------
    modeled : np.ndarray of float
        The modeled values.

    offsets : dict
        Key value pairs of `CalGroup` and offset values [mag].

    pos : dict
        The calibration positions.

    Returns
    -------
    np.ndarray
        The modeled values with applied offsets.
    """
    if offsets is not None:
        for name, offset in offsets.items():
            modeled[pos[name]] *= 10.0 ** -(0.4 * offset)

    return modeled


def slop(params, obs) -> float | np.ndarray | None:
    """
    Formats the slop according to data groups.

    Parameters
    ----------
    params : dict
        The dict returned from `Parameters.samples_to_dict`.

    obs : Observation
        The observational data.

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
    s = np.empty(obs.length)

    for group, pos in obs.groups.items():
        s[pos] = params.get(group).get('slop').get('slop')

    return s


def chi_squared(modeled, obs, slop=None) -> float:
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
            obs.as_arrays.values,
            obs.as_arrays.errors,
        )

    # Chi-squared for flux (uses slop)
    flux_mask = obs.flux_loc

    cs_flux = math_utils.chi_squared(
        modeled[flux_mask],
        obs.as_arrays.values[flux_mask],
        obs.as_arrays.errors[flux_mask],
        slop if isinstance(slop, float) else slop[flux_mask]
    )

    # Chi-squared for spectral indices (does not use slop)
    index_mask = obs.sindex_loc

    if not index_mask.any():
        return cs_flux

    cs_indices = math_utils.chi_squared(
        modeled[index_mask],
        obs.as_arrays.values[index_mask],
        obs.as_arrays.errors[index_mask],
    )

    # return combined chi-squared
    return cs_flux + cs_indices
