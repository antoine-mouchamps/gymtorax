import xarray as xr
from xarray import DataTree, Dataset
from numpy.typing import NDArray

def get_data(ds: Dataset, key: str) -> NDArray:
    """
    Get data from an xarray dataset.
    
    Args:
        ds (xr.Dataset): The xarray dataset to retrieve data from.
        key (str): The key of the data variable to retrieve.

    Raises:
        KeyError: If the specified key is not found in the dataset.

    Returns:
        NDArray: The data array corresponding to the specified key.
    """
    if key in ds:
        value = ds[key].to_numpy().tolist()[0]
        if(type(value) is list):
            return value
        else:
            return [value]
    else:
        raise KeyError(f"Key '{key}' not found in dataset.")


def get_dataset(datatree: DataTree) -> Dataset:
    """
    Get the dataset from a DataTree. The dataset is created by merging the
    `/profiles` and `/scalars` and getting rid of the `/numerics` one.

    Args:
        datatree (DataTree): The DataTree to extract the dataset from.

    Returns:
        Dataset: The merged dataset.
    """
    profiles = datatree["/profiles/"].ds
    scalars = datatree["/scalars/"].ds
    
    merged = xr.merge([profiles, scalars])
    merged = merged.drop_vars("drho_norm")
    
    return merged