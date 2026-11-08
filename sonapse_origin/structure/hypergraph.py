"""Higher-order Hypergraph Message Passing and matched graph baseline comparisons."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HypergraphMessagePassing:
    """Implements higher-order hypergraph communication:
    
        m_i = sum_{e in E, i in e} phi_e({x_j : j in e})
        
    where phi_e computes non-linear higher-order interactions across all nodes in hyperedge e.
    """

    def __init__(self, hyperedges: List[List[int]], node_dim: int = 4):
        self.hyperedges = hyperedges
        self.node_dim = node_dim

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes hypergraph message passing across all nodes."""
        num_nodes = X.shape[0]
        messages = np.zeros_like(X)

        for edge in self.hyperedges:
            if not edge:
                continue
            # Higher-order interaction: product or non-linear combination across member nodes
            edge_states = X[edge]  # shape: (order, node_dim)
            # Synergistic interaction: order-scaled non-linear aggregation
            order = len(edge)
            synergy = np.prod(np.tanh(edge_states), axis=0) * np.sqrt(order)
            mean_state = np.mean(edge_states, axis=0)
            combined_message = 0.5 * mean_state + 0.5 * synergy

            for node_idx in edge:
                messages[node_idx] += combined_message

        return messages


class MatchedGraphBaseline:
    """Parameter-matched standard pairwise graph baseline for controlled comparison."""

    def __init__(self, hyperedges: List[List[int]], num_nodes: int, node_dim: int = 4):
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        # Project hyperedges into pairwise clique adjacency
        self.adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        for edge in hyperedges:
            for u in edge:
                for v in edge:
                    if u != v:
                        self.adj[u, v] += 1.0 / max(1, len(edge) - 1)
        # Normalize adjacency
        row_sums = self.adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.adj /= row_sums

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Standard linear pairwise message passing: m_i = sum_j W_ij x_j."""
        return self.adj @ X
