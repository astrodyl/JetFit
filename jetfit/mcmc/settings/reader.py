from pathlib import Path

from jetfit.core.utilities.io.toml import TOMLReader
from jetfit.models.afterglow import models


class MCMCSettingsReader(TOMLReader):
    """
    Reader for MCMC Settings TOML config file.

    Attributes
    ----------
    num_walkers : float
        The number of walkers.

    burn_length : float
        The number of iterations to burn.

    run_length : float
        The number of iterations to run.

    model : str
        The model to use for the MCMC routine.
    """
    def __init__(self, path: str | Path, live_dangerously: bool = False):
        """
        Parameters
        ----------
        live_dangerously : bool, optional
            If ``True``, skips the validation process.
        """
        super().__init__(path)

        if not live_dangerously:
            self.validate()

        sampler = self.data.get('sampler')
        self.num_walkers = sampler.get('num_walkers')
        self.burn_length = sampler.get('burn_length')
        self.run_length = sampler.get('run_length')
        self.model = models.get(self.data.get('model'))

    def validate(self) -> None:
        """ Validates that the MCMC settings file is valid. """
        self.validate_sampler()
        self.validate_model()

    def validate_sampler(self) -> None:
        """ Validates that the sampler section is valid. """
        sampler = self.get_section('sampler')

        self.validate_value('burn_length', sampler.get('burn_length'), int)
        self.validate_value('run_length',  sampler.get('run_length'),  int)
        self.validate_value('num_walkers', sampler.get('num_walkers'), int)

    def validate_model(self) -> None:
        """ Validates that the model section is valid. """
        self.validate_value('model', self.get_section('model'), str)
