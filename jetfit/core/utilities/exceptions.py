

class EvidenceColumnError(Exception):
    """
    Exception for a missing column.
    """
    def __init__(self, name: str):
        """
        Parameters
        ----------
        name : str
            A row from the input evidence CSV.
        """
        super().__init__(f"Missing column '{name}'.")


class EvidenceTypeError(Exception):
    """
    Exception for receiving invalid evidence type when reading the input CSV.
    """
    def __init__(self, row):
        """
        Parameters
        ----------
        row : NamedTuple
            A row from the input evidence CSV.
        """
        super().__init__(
            f"Unrecognized evidence type: '{row.ValueType}'.\n"
            f"{row.Index}\t{row.ValueType}"
        )
        self.row_idx = row.Index
        self.type = row.ValueType


class EvidenceValueError(Exception):
    """
    Exception for receiving invalid evidence type when reading the input CSV.
    """
    def __init__(self, row):
        """
        Parameters
        ----------
        row : NamedTuple
            A row from the input evidence CSV.
        """
        super().__init__(
            f"Received invalid value in row {row.Index} of the evidence CSV."
        )
