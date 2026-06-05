"""A function which can be used for logging algorithm performance to the open-source DiscoHub dataset. There are a number of options which can be provided to the function."""

from pathlib import Path


def log(src: Path, scores: dict[str, float | str]) -> None:
    """Design structure:.

    API Key - either at input, or os env variable
    MetaData - Dict with options - only allowed entries





    Algorithms (in code) x
    Their performance on any datasets tested x
    Any error messages x
    Various types of metadata
    Date/Time
    LLM
    Agent
    Hardware
    Memory usage, power usage etc.
    Meta-Train or Meta-Test
    All the above is optional to store
    Any non-default licenses (default will be Apache, but people can select a noncommercial license e.g.).
    """
    return None
