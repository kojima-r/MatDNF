"""Test fixtures."""

import itertools
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy import typing as npt


def _sort_by_index_suffix(path: Path) -> int:
    """Get numerical index from a path stem."""
    return int(path.stem.split("_")[-1])


def zip_strict_nonempty[T: Iterator[Any]](*args: T) -> Iterator[T]:
    a, b = itertools.tee(zip(*args, strict=True))
    try:
        next(b)
    except StopIteration:
        raise ValueError("Array iterators are empty.")
    return a


def iter_matlab_arrays[T: np.generic](
    prefix: str,
    dtype: npt.DTypeLike,
    directory: Path = Path("tests/resources/mat_dnf"),
) -> Iterator[npt.NDArray[T]]:
    return (
        np.loadtxt(x, dtype=dtype, delimiter=",")
        for x in sorted(
            [
                y
                for y in directory.glob(f"*.csv")
                if re.match(f"{prefix}_[0-9]+", y.name)
            ],
            key=_sort_by_index_suffix,
        )
    )
