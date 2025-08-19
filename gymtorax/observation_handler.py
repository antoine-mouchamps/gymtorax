from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import xarray as xr
from .utils import get_dataset, get_data


class Observation:
    # Class-level attributes to be overridden by subclasses
    default_min: list = None  
    default_max: list = None
    variables: list[str]|None = None
    
    
    def __init__(self, selection: list[str]|None = None, custom_bounds: dict[str, dict[str, float]] = {}):
        min_bounds = self.default_min
        max_bounds = self.default_max
        for key, bounds in custom_bounds.items():
            if "min" in bounds.keys():
                min_bounds[self.variables.index(key)] = bounds["min"]
            if "max" in bounds.keys():
                max_bounds[self.variables.index(key)] = bounds["max"]
            if key not in self.variables:
                raise KeyError(f"Variable {key} is not a StateProfile variable!")
        
        self._min = min_bounds
        self._max = max_bounds
     
        if selection is not None:
            for var in selection:
                if var not in self.variables:
                    raise KeyError(f"Variable {var} is not a StateProfile variable!")
            self.variables = selection


    @property
    def min(self) -> list:
        """Minimum bounds for this state."""
        return self._min
    
    @property
    def max(self) -> list:
        """Maximum bounds for this state."""
        return self._max

    def update(self):
        pass

    def get_values_from_data(self, data: xr.DataTree)->list:
        if self.variables is None:
            raise ValueError("State variables are not defined.")
        vals = []
        for var in self.variables:
            vals.extend(self.get_val_from_data(data, var))

        return vals

    @abstractmethod
    def get_val_from_data(self, data: xr.DataTree, key: str)->Any:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(value={self.value}, min={self.min[0]}, max={self.max[0]})"


class ObservationHandler:
    def __init__(self, observations: list[Observation] = []):
        self.observations: list[Observation] = observations
        self.variables: list[str] = []    
        
        for observation in self.observations:
            self.variables.extend(observation.variables)
    
    def get_observations(self) -> list[Observation]:
        """Get the list of observations."""
        return self.observations

    def get_observation_values(self, data: xr.Dataset):
        values = []
        
        if len(self.variables) == 0:
            for key in data.keys():
                values.extend(get_data(data, key))
        else:
            for variable in self.variables:
                values.extend(get_data(data, variable))

        return np.array(values)

if __name__ == "__main__":
    # Example usage
    from torax._src.output_tools import output

    dt = output.load_state_file('gymtorax/output.nc')
    ds = get_dataset(dt)

    dim = 0
    # def print_dtree(dt, indent=0):
    #     prefix = "  " * indent
    #     print(f"{prefix}{dt.name}/")
    #     if dt.has_data:
    #         for var in dt.ds.variables:
    #             print(f"{prefix}  {var} : {dt.ds[var].dims} -> {dt.ds[var].shape}")
    #     for child in dt.children.values():
    #         print_dtree(child, indent + 1)

    # print_dtree(dt)
    
    oh = ObservationHandler()
    os = oh.get_observation(ds)
    
    # for key in ds.keys():
    #     val = get_data(ds, key)
    #     dim += val.size
        
    # print(dim)
