from __future__ import annotations

from typing_extensions import Self
from queue import LifoQueue

import numpy as np


class Node:
    _name: str
    distance = np.inf
    _child_nodes: list[Self]

    def __init__(self, name: str):
        self._name = name
        self._child_nodes = []

    @property
    def name(self):
        return self._name

    @property
    def children(self):
        return self._child_nodes

    def append_child(self, child: Self):
        self._child_nodes.append(child)

class DWG:
    root: Node
    _edges: dict[tuple[Node, Node], int]
    _nodes: dict[str, Node]

    def __init__(self, root: Node) -> None:
        self.root = root
        self._edges = dict()
        self._nodes = dict()

    @property
    def edges(self):
        return self._edges

    @property
    def nodes(self):
        return self._nodes

    def weight(self, u: Node, v: Node) -> int | None:
        return self._edges.get((u, v), None)

    def add_node(self, node: Node) -> None:
        self._nodes[node.name] = node

    def add_edge(self, u: Node, v: Node, weight: int) -> None:
        """
        Adds an edge, between two nodes.
        :param u: The node with an outgoing edge.
        :param v: The node with an incoming edge.
        """
        if u.name not in self._nodes:
            self.add_node(u)
        if v.name not in self._nodes:
            self.add_node(v)

        u.append_child(v)

        # adding the weight to the edge
        self._edges[(u, v)] = weight

    def get(self, node_name: str) -> Node | None:
        """
        Returns a node with a given name, if it exists, otherwise returns None.
        """
        try:
            return self._nodes[node_name]
        except KeyError:
            return None

    def get_or_new(self, name: str) -> Node:
        """
        Returns a node with a given name, if it exists, otherwise creates a new node, and returns that.
        """
        if node := self.get(name) is None:
            node = Node(name)
            self.add_node(node)
            return node
        else:
            return node

    def dfs(self, name: str) -> Node | None:
        """
        performs a dfs to retrieve a node
        :param name: the name of the node
        :return: the node or None if the node is not found
        """
        if name == self.root.name:
            return self.root
        stack = LifoQueue()
        discovered = []
        stack.put(self.root)
        while not stack.empty():
            vertex = stack.get_nowait()
            if vertex.name not in discovered:
                discovered.append(vertex.name)
                for child in vertex.children:
                    if child.name == name:
                        return child
                    else:
                        stack.put(child)

        return None
