import numpy as np


def cartesian_product(*arrays):
    """Compute cartesian product of arrays
    with different shapes in an efficient manner.

    Args:
        arrays: each array shoud be rank 2 with shape (N_i, d_i).
        inds: indices for each array, should be rank 1.

    Returns:
        Cartesian product of arrays with shape (N_1, N_2, ..., N_n, sum(d_i)).
    """
    _arrays = [*map(lambda x: x if x.ndim > 1 else x[:, np.newaxis], arrays)]
    d = [*map(lambda x: x.shape[-1], _arrays)]
    ls = [*map(len, _arrays)]
    inds = [*map(np.arange, ls)]

    dtype = np.result_type(*_arrays)
    arr = np.empty(ls + [sum(d)], dtype=dtype)

    for i, ind in enumerate(np.ix_(*inds)):
        arr[..., sum(d[:i]) : sum(d[: i + 1])] = _arrays[i][ind]
    return arr
