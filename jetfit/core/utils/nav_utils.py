import os
from pathlib import Path


# <editor-fold desc="Project Navigation">
def get_project_root() -> Path:
    """
    Returns the project root directory.

    Returns
    -------
    Path
        The project root directory.
    """
    return Path(__file__).parent.parent.parent.parent


def get_test_path() -> Path:
    """
    Returns the project test directory.

    Returns
    -------
    Path
        The project test directory.
    """
    return get_project_root() / 'test'


def get_models_path() -> Path:
    """
    Returns the project models directory.

    Returns
    -------
    Path
        The project models directory.
    """
    return get_project_root() / 'jetfit' / 'models'


def get_results_path() -> Path:
    """
    Returns the project results directory.

    Returns
    -------
    Path
        The project results directory.
    """
    return get_project_root() / 'jetfit' / 'results'


def get_resource_path() -> Path:
    """
    Returns the project resource directory.

    Returns
    -------
    Path
        The project resource directory.
    """
    return get_project_root() / 'jetfit' / 'resources'


def get_event_path(category: str, event: str) -> Path:
    """
    Returns the path to the event's resource directory.

    Parameters
    ----------
    category : str

    event : str
        Name of event directory in resources

    Returns
    -------
    Path
        The event resource path.
    """
    return get_resource_path() / category / event


def get_input_csv_path(category: str, event: str) -> Path:
    """
    Returns the path to the input csv file.

    Parameters
    ----------
    category : str

    event : str
        Name of event directory in resources

    Notes
    -----
    The input CSV file should be named the same as the event directory.

    Returns
    -------
    Path
        The input CSV file path.
    """
    return get_event_path(category, event) / f'{event}.csv'


def get_mcmc_settings_path() -> Path:
    """ Returns the path to the MCMC settings. """
    return get_project_root() / 'jetfit' / 'mcmc' / 'settings' / 'settings.toml'


def get_boosted_fireball_path():
    """ Returns the path to the boosted fireball root directory. """
    return get_project_root() / 'jetfit' / 'models' / 'afterglow' / 'boosted_fireball'


def get_boosted_fireball_params_path() -> Path:
    """ Returns the path to the boosted fireball params config. """
    return get_boosted_fireball_path() / 'parameters' / 'defaults.toml'


def get_hydro_sim_table_path() -> Path:
    """
    Returns the path to the hydrodynamic simulation table.

    Returns
    -------
    Path
        The input hydrodynamic simulation table.
    """
    return get_boosted_fireball_path() / 'hydro_sim' / 'hydro_sim_new.h5'


def get_valid_model_names() -> list[str]:
    """
    Returns a list of subdirectory names in ``get_models_path()``.

    Returns
    -------
    list of str
        The list of valid model names.
    """
    valid = []
    for entry in os.scandir(get_models_path() / 'afterglow'):
        if entry.is_dir():
            valid.append(entry.name)
    return valid
#</editor-fold>
