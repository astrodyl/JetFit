from jetfit.core.defns.enums import ScaleType
from jetfit.core.utils import maths, paths
from jetfit.mcmc.parameters import priors


class MCMCParameter:
    """
    A parameter to be used with MCMC.

    Attributes
    ----------
    name : str
        The name of the parameter.

    scale : `jetfit.core.enums.ScaleType`
        The scale of the parameter.
    """
    def __init__(
            self,
            name: str,
            scale: ScaleType
    ):
        self.name = name
        self.scale = scale


class MCMCFixedParameter(MCMCParameter):
    """
    A fixed parameter to be used within a model for MCMC.

    Attributes
    ----------
    value : float
        The fixed value of the parameter.
    """
    def __init__(
            self,
            name: str,
            value: float,
            scale: ScaleType
    ):
        super().__init__(name, scale)
        self.value = value

    @classmethod
    def from_dict(cls, name: str, params: dict):
        """
        Instantiates the class from a dictionary.

        Parameters
        ----------
        name : str
            The name of the parameter.

        params : dict
            The class attributes and values.

        Returns
        -------
        MCMCFixedParameter
            Instantiated from `params`.
        """
        if not paths.is_expected_type(params.get('value'), float):
            raise ValueError(f'Received unexpected value information for'
                             f'{name}.')

        if not paths.is_expected_type(params.get('scale'), str):
            raise ValueError(f'Received unexpected scale information for'
                             f'{name}.')

        return cls(
            name,
            params.get('value'),
            ScaleType(params.get('scale'))
        )

    def get_value(self, scale: ScaleType | str) -> float | None:
        """
        Returns ``value`` in ``scale``.

        Parameters
        ----------
        scale : ScaleType or str
            The scale of the parameter to return.

        Returns
        -------
        float
            The value in the specified scale.
        """
        return maths.to_scale(self.value, self.scale, scale)


class MCMCFittingParameter(MCMCParameter):
    """
    A free parameter to be sampled with the MCMC routine.

    Attributes
    ----------
    prior : `jetfit.core.enums.Prior`
        The prior probability distribution.
    """
    def __init__(
            self,
            name: str,
            scale: ScaleType,
            prior
    ):
        MCMCParameter.__init__(self, name, scale)
        self.prior = prior

    @classmethod
    def from_dict(cls, name: str, params: dict):
        """
        Instantiates the class from a dictionary.

        Parameters
        ----------
        name : str
            The name of the parameter.

        params : dict
            The class attributes and values.

        Returns
        -------
        MCMCFittingParameter
            Instantiated from `params`.
        """
        if not paths.is_expected_type(params.get('prior'), dict):
            raise ValueError(f'Received unexpected prior information for'
                             f'{name}.')

        if not paths.is_expected_type(params.get('scale'), str):
            raise ValueError(f'Received unexpected scale information for '
                             f'{name}.')

        return cls(
            name,
            ScaleType(params.get('scale')),
            priors.prior_factory(params.get('prior'))
        )
