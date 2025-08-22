from torax._src.output_tools.output import concat_datatrees
from functools import reduce
from xarray import DataTree

def merge_history_list(list_dataTree : list[DataTree]):
    """Merge all DataTrees in history_list into a single DataTree along time.
    args:
        list_dataTree: A list of DataTree objects to merge.
    returns:
        A merged DataTree object or None if the input list is empty.
    """
    if not list_dataTree or len(list_dataTree) <= 1:
        return None
    merged = reduce(concat_datatrees, list_dataTree)
    return merged
