import tomllib
from pathlib import Path


def read(path: str | Path) -> dict:
    """ """
    with open(path, "rb") as f:
        return tomllib.load(f)
