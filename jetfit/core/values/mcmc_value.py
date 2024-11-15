from jetfit.core.defns.enums import ParameterGroup, ScaleType
from jetfit.core.utilities import utils, maths
from jetfit.mcmc import priors


class MCMCParameter:
    """
    A parameter to be used with MCMC.

    Attributes
    ----------
    name : str
        The name of the parameter.

    group : `jetfit.core.enums.ParameterGroup`
        The grouping of the parameter.

    scale : `jetfit.core.enums.ScaleType`
        The scale of the parameter.
    """
    def __init__(self, name: str, group: ParameterGroup, scale: ScaleType):
        self.name = name
        self.group = group
        self.scale = scale


class MCMCFixedParameter(MCMCParameter):
    """
    A fixed parameter to be used within a model for MCMC.

    Attributes
    ----------
    value : float
        The fixed value of the parameter.
    """
    def __init__(self, name: str, value: float, group: ParameterGroup,
                 scale: ScaleType):

        super().__init__(name, group, scale)
        self.value = value

    @property
    def fit(self) -> bool:
        """ Returns bool indicating that the value is fixed. """
        return False

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
        if not utils.is_expected_type(params.get('value'), float):
            raise ValueError(f'Received unexpected value information for'
                             f'{name}.')

        if not utils.is_expected_type(params.get('group'), str):
            raise ValueError(f'Received unexpected group information for'
                             f'{name}.')

        if not utils.is_expected_type(params.get('scale'), str):
            raise ValueError(f'Received unexpected scale information for'
                             f'{name}.')

        return cls(
            name,
            params.get('value'),
            params.get('group'),
            ScaleType(params.get('scale'))
        )

    def get_value(self, scale: ScaleType | str) -> float | None:
        """
        Returns the value in the specified scale.

        Parameters
        ----------
        scale : ScaleType or str
            The scale of the parameter to return

        Returns
        -------
        float
            Value in the specified scale
        """
        return maths.to_scale(self.value, self.scale, scale)


class MCMCFittingParameter(MCMCParameter):
    """
    A parameter to be used with MCMC for fitting.

    Attributes
    ----------
    prior : `jetfit.core.enums.Prior`
        Prior probability distribution.

    Raises
    ------
        ValueError
            If `wander` is `False` and `region` or `default` is `None`.
    """
    def __init__(self, name: str, group: ParameterGroup, scale: ScaleType,
                 prior):

        MCMCParameter.__init__(self, name, group, scale)
        self.prior = prior

    @property
    def fit(self) -> bool:
        """ Returns bool indicating that the value should be fit to. """
        return True

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
        if not utils.is_expected_type(params.get('prior'), dict):
            raise ValueError(f'Received unexpected prior information for'
                             f'{name}.')

        if not utils.is_expected_type(params.get('group'), str):
            raise ValueError(f'Received unexpected group information for'
                             f'{name}.')

        if not utils.is_expected_type(params.get('scale'), str):
            raise ValueError(f'Received unexpected scale information for'
                             f'{name}.')

        return cls(
            name,
            params.get('group'),
            ScaleType(params.get('scale')),
            priors.prior_factory(params.get('prior'))
        )
