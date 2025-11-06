import pytest
import numpy as np

from mercury.graph.core import Graph


def test_register_node_feature():
    graph = Graph()
    graph.register_node_feature(name="feat", dim = 1, init_value=0.)
    print(graph.node_features)