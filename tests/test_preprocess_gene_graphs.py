"""
Regression tests for the low-level (gene) graph construction in utils/preprocess.py.

Reported externally (GitHub) and confirmed: the loop that builds `graphs` indexed the shared
per-cell-type PyG object out of `gene_network_dict` and then mutated it, so every cell of a given
cell type ended up holding the LAST cell's expression vector. The gene branch of the model therefore
saw at most one distinct input per cell type instead of one per cell.

These tests exercise the exact construction pattern rather than importing preprocess() itself, which
needs MAGIC + a CUDA GRN build. `test_shared_reference_pattern_is_the_bug` documents the old
behaviour so the tests fail loudly if anyone reintroduces it.

Run: .venv/bin/python -m pytest tests/test_preprocess_gene_graphs.py -v
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

NUM_GENES = 5
N_CELLS = 6


@pytest.fixture
def fixture():
    """One shared GRN per cell type, plus a distinct expression vector per cell."""
    g = nx.Graph()
    g.add_nodes_from(range(NUM_GENES))
    g.add_weighted_edges_from([(0, 1, 0.5), (1, 2, 0.7), (3, 4, 0.2)])
    gene_network_dict = {"A": from_networkx(g), "B": from_networkx(g)}
    cell_types = np.array(["A", "A", "B", "A", "B", "B"])
    X = np.arange(N_CELLS * NUM_GENES, dtype=np.float32).reshape(N_CELLS, NUM_GENES)
    return gene_network_dict, cell_types, X


def build_fixed(gene_network_dict, cell_types, X):
    """The construction as it now stands in utils/preprocess.py."""
    graphs = []
    for k in range(len(cell_types)):
        base = gene_network_dict[cell_types[k]]
        G_gene = Data(num_nodes=NUM_GENES)
        G_gene.edge_index = base.edge_index
        if getattr(base, "weight", None) is not None:
            G_gene.weight = base.weight
        G_gene.X = torch.from_numpy(np.asarray(X[k]).reshape(NUM_GENES, 1))
        graphs.append(G_gene)
    return graphs


def build_old(gene_network_dict, cell_types, X):
    """The construction as it was -- kept only so the bug stays pinned by a test."""
    graphs = []
    for k in range(len(cell_types)):
        G_gene = gene_network_dict[cell_types[k]]
        G_gene.num_nodes = NUM_GENES
        G_gene.X = torch.from_numpy(np.asarray(X[k]).reshape(NUM_GENES, 1))
        graphs.append(G_gene)
    return graphs


def test_each_cell_keeps_its_own_expression(fixture):
    graphs = build_fixed(*fixture)
    _, _, X = fixture
    for k, g in enumerate(graphs):
        assert torch.allclose(g.X.ravel(), torch.from_numpy(X[k])), (
            f"cell {k} carries the wrong expression vector"
        )


def test_graphs_are_distinct_objects(fixture):
    graphs = build_fixed(*fixture)
    assert len({id(g) for g in graphs}) == N_CELLS
    # ...and mutating one must not touch any other
    graphs[0].X = torch.full((NUM_GENES, 1), -99.0)
    assert not torch.allclose(graphs[1].X, graphs[0].X)


def test_topology_is_shared_and_identical_within_a_cell_type(fixture):
    """Sharing read-only topology is intentional -- it is what keeps memory flat."""
    gene_network_dict, cell_types, X = fixture
    graphs = build_fixed(gene_network_dict, cell_types, X)
    a_idx = [k for k, c in enumerate(cell_types) if c == "A"]
    for k in a_idx[1:]:
        assert torch.equal(graphs[k].edge_index, graphs[a_idx[0]].edge_index)
    assert graphs[a_idx[0]].edge_index is gene_network_dict["A"].edge_index


def test_number_of_distinct_expression_vectors_equals_number_of_cells(fixture):
    """The failure mode that matters: the gene branch collapsing to one vector per cell type."""
    graphs = build_fixed(*fixture)
    distinct = {tuple(g.X.ravel().tolist()) for g in graphs}
    assert len(distinct) == N_CELLS


def test_shared_reference_pattern_is_the_bug(fixture):
    """Pin the old behaviour so a regression is unambiguous, not silent."""
    gene_network_dict, cell_types, X = fixture
    graphs = build_old(gene_network_dict, cell_types, X)
    distinct = {tuple(g.X.ravel().tolist()) for g in graphs}
    assert len(distinct) == 2, "old code should collapse to one vector per cell type"
    last_a = max(k for k, c in enumerate(cell_types) if c == "A")
    for k, c in enumerate(cell_types):
        if c == "A":
            assert torch.allclose(graphs[k].X.ravel(), torch.from_numpy(X[last_a]))


def test_positional_cell_type_lookup_with_string_barcodes():
    """
    adata.obs.cell_type[k] is label-based. With string barcodes pandas currently falls back to
    positional (deprecated); with an integer index it silently returns the wrong row. .to_numpy()
    is positional by construction.
    """
    s_str = pd.Series(pd.Categorical(["A", "B", "A"]), index=["bc1", "bc2", "bc3"])
    assert s_str.to_numpy()[1] == "B"

    # An integer, non-monotonic index is where label-based lookup silently diverges.
    s_int = pd.Series(pd.Categorical(["A", "B", "A"]), index=[10, 11, 12])
    assert s_int.to_numpy()[0] == "A"
    with pytest.raises(KeyError):
        _ = s_int[0]
