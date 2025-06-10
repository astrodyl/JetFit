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
    return get_models_path() / 'boosted'


def get_hydro_sim_table_path() -> Path:
    """
    Returns the path to the hydrodynamic simulation table.

    Returns
    -------
    Path
        The input hydrodynamic simulation table.
    """
    return get_models_path() / 'rsrcs' / 'hydro_sim_new.h5'
#</editor-fold>
