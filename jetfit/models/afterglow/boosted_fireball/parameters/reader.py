from pathlib import Path

from jetfit.core.utilities.io.toml import TOMLReader
from jetfit.mcmc.parameters.parameters import MCMCFixedParameter
from jetfit.mcmc.parameters.parameters import MCMCFittingParameter
from jetfit.models.afterglow.boosted_fireball.parameters.parameters import BFModelParams


class BFParamsReader(TOMLReader):
    """
    Reads and validates a Boosted Fireball Parameters file.
    """
    def __init__(self, path: str | Path, live_dangerously: bool = False):
        super().__init__(path)

        if not live_dangerously:
            self.verify()

        self.fitting = [
            MCMCFittingParameter.from_dict(k, v) for k, v
            in self.data.items() if 'prior' in v
        ]

        self.fixed = [
            MCMCFixedParameter.from_dict(k, v) for k, v
            in self.data.items() if 'value' in v
        ]

    def verify(self) -> None:
        """
        Verifies that every Boosted Fireball parameter is defined in the file.

        Raises
        ------
        ValueError
            If any of the parameters are missing from the file.
        """
        d = {key: 1.0 for key in self.data}

        if m := BFModelParams.from_dict(d).missing:
            raise ValueError(
                f'The following Boosted Fireball parameters are not '
                f'defined: {' '.join(m)}. Please add them to the '
                f'{self.path} file.'
            )
