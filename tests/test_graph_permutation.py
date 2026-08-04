"""
Regression test for the node-relabelling bug in utils.dataloader.shuffle_node_indices.

The bug: features were permuted by `perm` (so new slot i holds old node perm[i], i.e. old->new is
perm^-1) while edge_index was relabelled by `perm`. The effective adjacency became the true graph
relabelled by perm^2 — same degree sequence, but edges connecting random pairs of cells. Because
`create_dataloader(..., permute=True)` is the DEFAULT and is what training used
(main_ddp.py -> create_dataloader_ddp), every pretraining run learned on a SCRAMBLED spatial graph.

These tests assert the relabelling is an isomorphism, which is the only thing it was ever meant to be.

Run: .venv/bin/python tests/test_graph_permutation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.data import Data

from utils.dataloader import shuffle_node_indices


def _path_graph(n=64):
    """Cells on a line at x=0..n-1; every true edge has length exactly 1."""
    coords = torch.arange(n, dtype=torch.float).reshape(n, 1)
    ei = torch.stack([torch.arange(n - 1), torch.arange(1, n)], dim=0)
    ei = torch.cat([ei, ei.flip(0)], dim=1)  # undirected
    d = Data(num_nodes=n)
    d.X = coords
    d.edge_index = ei
    return d


def test_edge_lengths_preserved():
    """Geometry must survive relabelling: mean |edge length| stays 1.0."""
    torch.manual_seed(0)
    for trial in range(20):
        d = _path_graph()
        before = (d.X[d.edge_index[0]] - d.X[d.edge_index[1]]).abs().mean().item()
        d, perm = shuffle_node_indices(d)
        after = (d.X[d.edge_index[0]] - d.X[d.edge_index[1]]).abs().mean().item()
        assert abs(before - after) < 1e-6, (
            f"trial {trial}: edge length changed {before:.4f} -> {after:.4f}; "
            "node relabelling is not an isomorphism"
        )
    print("PASS  edge lengths preserved under permutation (20 trials)")


def test_neighbour_sets_preserved():
    """The neighbour SET of every cell must be unchanged, up to relabelling."""
    torch.manual_seed(1)
    n = 32
    d = _path_graph(n)
    orig = {int(v): set() for v in range(n)}
    for a, b in d.edge_index.t().tolist():
        orig[a].add(b)

    d, perm = shuffle_node_indices(d)
    # Identity must be read AFTER the shuffle: coords encode cell id, so d.X[s] is the id of the
    # cell now sitting in slot s.
    ids = d.X.squeeze(-1)
    new = {}
    for a, b in d.edge_index.t().tolist():
        new.setdefault(int(ids[a].item()), set()).add(int(ids[b].item()))

    for cell, nbrs in orig.items():
        assert new.get(cell, set()) == nbrs, (
            f"cell {cell}: neighbours changed {nbrs} -> {new.get(cell, set())}"
        )
    print("PASS  neighbour sets preserved under permutation")


def test_low_level_mapping_consistent():
    """
    create_dataloader indexes low-level graphs with `perm[partition_node_indices]`.
    Partition indices are NEW slots; new slot s holds old cell perm[s]; so perm[...] is correct.
    Assert the feature sitting at slot s really is old cell perm[s]'s feature.
    """
    torch.manual_seed(2)
    d = _path_graph(48)
    before = d.X.clone()
    d, perm = shuffle_node_indices(d)
    assert torch.equal(d.X, before[perm]), "X at slot s is not old cell perm[s]"
    print("PASS  low-level graph indexing (perm[partition_idx]) is consistent")


if __name__ == "__main__":
    test_edge_lengths_preserved()
    test_neighbour_sets_preserved()
    test_low_level_mapping_consistent()
    print("\nAll permutation tests passed.")
