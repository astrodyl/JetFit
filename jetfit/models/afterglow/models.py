from jetfit.core.utils import paths
from jetfit.models.afterglow.boosted_fireball.boosted_fireball import BoostedFireball


def get(name: str):
    """
    Finds the model class by name and returns it.

    Parameters
    ----------
    name : str
        The name of the model contained in `jetfit.models.afterglow`.

    Raises
    ------
    ValueError
        If ``name`` is not in ``paths.get_valid_model_names()``.
    """
    if name not in paths.get_valid_model_names():
        raise ValueError(
            f'Invalid model name: {name}. Have you added the the model to '
            f'{paths.get_models_path()}?'
        )

    if name == 'boosted_fireball':
        return BoostedFireball()
