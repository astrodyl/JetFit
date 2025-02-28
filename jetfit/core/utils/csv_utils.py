from pathlib import Path
import pandas as pd


SCHEMA = {
    'required': {
        'Time': {"type": float},
        'TimeUnits': {"type": str},
        'Value': {"type": float},
        'ValueType': {"type": float},
        'ValueLower': {"type": float},
        'ValueUpper': {"type": float},
        'ValueUnits': {"type": str},
        'Model': {'type': str}
    },
    'conditional': {
        'Frequency': {'type': float, 'required_by': {'ValueType': 'SpectralFlux'}},
        'FrequencyLower': {'type': float, 'required_by': {'ValueType': 'IntegratedFlux'}},
        'FrequencyUpper': {'type': float, 'required_by': {'ValueType': 'IntegratedFlux'}},
        'FrequencyUnits': {'type': float, 'required_by': {'ValueType': 'IntegratedFlux'}},
    }
}


class CSVReader:
    """
    Reads an evidence CSV and stores it in a pandas DataFrame.

    Attributes
    ----------
    df : pd.DataFrame
        Pandas representation of the CSV file.
    """
    schema = SCHEMA

    def __init__(self, path: str | Path, live_dangerously: bool = False):
        """

        Parameters
        ----------
        path : str | Path
            Location to the csv file.

        live_dangerously : bool, optional
            If `True`, skips verification of CSV.
        """
        self.df = pd.read_csv(path)

        if not live_dangerously:
            self.validate()

    def rows(self):
        """"""
        return self.df.itertuples(name='Evidence')

    # def validate(self) -> None:
    #     """ Checks that the CSV contains the required valid information. """
    #     self.validate_headers()
    #     self.validate_rows()
    #
    # def validate_headers(self) -> None:
    #     """
    #     Checks that the required headers are present.
    #
    #     Does not check if the CSV contains any column headers that are not
    #     required or conditionally required since it does not affect the
    #     read-in. Users are allowed to have a CSV containing anything that
    #     they want so long as the required data are present.
    #
    #     Raises
    #     ------
    #     """
    #     for header in self.schema.get('required'):
    #         if header not in self.df.columns:
    #             raise ValueError('Missing header')
    #
    # def validate_rows(self) -> None:
    #     """
    #     Checks that every row has the required valid values.
    #
    #     Raises
    #     ------
    #     """
    #     for row in self.rows():
    #         self.validate_required_values(row)
    #
    #         match row.ValueType.lower():  # type: ignore
    #             case FluxType.INTEGRATED.value:
    #                 self.validate_value(row, 'FrequencyLower', float)
    #                 self.validate_value(row, 'FrequencyUpper', float)
    #                 self.validate_value(row, 'FrequencyUnits', str)
    #
    #             case FluxType.SPECTRAL.value:
    #                 self.validate_value(row, 'Frequency', float)
    #                 self.validate_value(row, 'FrequencyUnits', str)
    #
    #             case IndexType.SPECTRAL.value:
    #                 pass
    #
    #             case _:
    #                 raise TypeError(
    #                     f"Received invalid value in row {row.Index}."
    #                 )
    #
    # def validate_required_values(self, row) -> None:
    #     """
    #     Checks that the required values are present and valid.
    #
    #     Parameters
    #     ----------
    #     row : NamedTuple
    #
    #     """
    #     self.validate_value(row, 'Time', float)
    #     self.validate_value(row, 'TimeUnits', str)
    #     self.validate_value(row, 'Value', float)
    #     self.validate_value(row, 'ValueLower', float)
    #     self.validate_value(row, 'ValueUpper', float)
    #     self.validate_value(row, 'ValueUnits', str)
    #     self.validate_value(row, 'ValueType', str)
    #     self.validate_value(row, 'Model', str)
    #
    # @staticmethod
    # def validate_value(row, name: str, expected_type) -> None:
    #     """
    #
    #     Parameters
    #     ----------
    #
    #     Raises
    #     ------
    #     """
    #     if (value := getattr(row, name)) is None:
    #         raise TypeError(
    #             f"Received invalid value in row {row.Index}."
    #         )
    #
    #     if not isinstance(value, expected_type):
    #         raise TypeError("Read in error")
    #
    #     return value
