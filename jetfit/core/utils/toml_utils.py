import tomllib
from pathlib import Path


class TOMLReader:
    def __init__(self, path: str | Path):
        self.path = path
        self.data = self.read()

    def read(self) -> dict:
        """
        Opens the TOML file at ``path``.

        Returns
        -------
        dict
            A dictionary of TOML data.
        """
        with open(self.path, "rb") as f:
            return tomllib.load(f)

    @staticmethod
    def validate_value(name: str, value, expected_type) -> None:
        """

        Raises
        ------
        TypeError
            If encounters unexpected value.
        """
        if not isinstance(value, expected_type):
            raise TypeError(f'Received unexpected value for {name}.')

    def get_section(self, section: str, optional: bool = False) -> dict | None:
        """
        Checks that the section exists and returns it if it does.

        Parameters
        ----------
        section : str
            The TOML section name.

        optional : bool, optional, default=False
            If ``True``, does not raise an exception if section does not
            exist. Instead, returns ``None``.

        Returns
        -------
        dict
            The section dictionary.

        Raises
        ------
        ValueError
            If the section does not exist and optional is ``False``.
        """
        if (data := self.data.get(section, None)) is None and not optional:
            raise ValueError(
                f'{self.path} does not contain a {section} section.'
            )

        return data
