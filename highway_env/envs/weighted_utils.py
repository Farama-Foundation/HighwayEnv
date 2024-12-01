from abc import ABC
import numpy as np

class WeightedUtils(ABC):
    np_random: np.random.Generator
    def get_random_edge_from(self, edge_list: list[tuple[str, str, int]]) -> tuple[str, str, int]:
        edge = self.np_random.choice(edge_list)
        return tuple((edge[0], edge[1], 0))
