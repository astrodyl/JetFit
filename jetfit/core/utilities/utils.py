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


def get_resource_path() -> Path:
    """
    Returns the project resource directory.

    Returns
    -------
    Path
        The project resource directory.
    """
    return get_project_root() / 'jetfit' / 'resources'


def get_input_csv_path(event: str) -> Path:
    """
    Returns the path to the input csv file.

    Parameters
    ----------
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
    return get_resource_path() / 'grbs' / event / f'{event}.csv'


def get_hydro_sim_table_path() -> Path:
    """
    Returns the path to the hydrodynamic simulation table.

    Notes
    -----
    Describe the table here. and reference paper.

    Returns
    -------
    Path
        The input hydrodynamic simulation table.
    """
    return get_resource_path() / 'tables' / 'Table_new.h5'


def get_mcmc_settings_path() -> Path:
    """ """
    return get_project_root() / 'jetfit' / 'mcmc' / 'config' / 'settings.toml'


def get_mcmc_params_path() -> Path:
    """ """
    return get_project_root() / 'jetfit' / 'mcmc' / 'config' / 'parameters.toml'
#</editor-fold>


# <editor-fold desc="Validation">
def is_expected_type(value, expected) -> bool:
    """
    Determines if the value is not `None` the expected type.

    Parameters
    ----------
    value : any
        The value to check.

    expected : type
        The expected type.

    Returns
    -------
    bool
        `True` if the value is not `None` the expected type.
    """
    return value is not None and isinstance(value, expected)
# </editor-fold>