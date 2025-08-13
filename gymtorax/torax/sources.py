from dataclasses import dataclass


@dataclass
class Bounds:
    min: float
    max: float

@dataclass
class SourceBounds:
    total: Bounds
    loc: Bounds
    width: Bounds
    
def expand_sources(ES_k_bounds:list[SourceBounds]) -> list[Bounds]:
    """Transform the list of SourceBounds (tuples of Bounds) into the corresponding
    list of bounds (deleting the SourceBounds wrapper)

    Returns
    -------
    list
        The expanded action space for the sources.
    """
    return [bounds for source in ES_k_bounds for bounds in (source.total, source.loc, source.width)]
    