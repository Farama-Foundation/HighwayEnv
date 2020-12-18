from typing import Union, Sequence, Tuple, List
import numpy as np

Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[np.ndarray,
                 Tuple[Vector, Vector],
                 Tuple[Matrix, Matrix],
                 Tuple[float, float],
                 List[Vector],
                 List[Matrix],
                 List[float]]