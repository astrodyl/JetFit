import pandas as pd


def read(path: str):
    """ Reads a csv at the path and returns a pd.DataFrame

    :param path: path to csv
    """
    return pd.read_csv(path)


def df_to_dict(df: pd.DataFrame) -> dict:
    """ Converts a pd.DataFrame to a dictionary.

    :param df: pd.DataFrame
    :return: dictionary with key, value from df
    """
    return {'times': df['Times'].values,
            'time_bounds': df['TimeBnds'].values,
            'fluxes': df['Fluxes'].values,
            'flux_errors': df['FluxErrs'].values,
            'frequencies': df['Freqs'].values}