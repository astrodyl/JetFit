from pathlib import Path

from jetfit.core.utils.toml_utils import TOMLReader
from jetfit.mcmc.parameters.parameters import MCMCFixedParameter
from jetfit.mcmc.parameters.parameters import MCMCFittingParameter


class BFParamsReader(TOMLReader):
    """
    Reads and validates a Boosted Fireball Parameters file.
    """
    def __init__(self, path: str | Path, live_dangerously: bool = False):
        super().__init__(path)

        self.fitting = [
            MCMCFittingParameter.from_dict(k, v) for k, v
            in self.data.items() if 'prior' in v
        ]

        self.fixed = [
            MCMCFixedParameter.from_dict(k, v) for k, v
            in self.data.items() if 'value' in v
        ]
