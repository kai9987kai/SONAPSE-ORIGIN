"""Architecture Genome specification and genetic operators for Structure Forge."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TopologyFamily(str, Enum):
    FEEDFORWARD = "feedforward"
    RECURRENT = "recurrent"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    HIERARCHICAL_MODULAR = "hierarchical_modular"
    RICH_CLUB = "rich_club"
    MULTILAYER_CONNECTOME = "multilayer_connectome"
    DYNAMIC_HYPERGRAPH = "dynamic_hypergraph"
    SPATIAL_NEURAL_FIELD = "spatial_neural_field"
    EVOLVING_COMMUNITY = "evolving_community"
    COUPLED_OSCILLATOR = "coupled_oscillator"
    HYBRID_NEURAL_SYMBOLIC = "hybrid_neural_symbolic"
    GLOBAL_WORKSPACE = "global_workspace"
    LOCAL_WORKSPACE_ONLY = "local_workspace_only"
    COMPETING_REGULATORY = "competing_regulatory"
    DYNAMIC_REPEATED_REWIRING = "dynamic_rewiring"


class CommunicationMode(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    COMPLEX_VALUED = "complex_valued"
    PHASE_CODED = "phase_coded"
    SPIKE_TIMING = "spike_timing"
    WAVE_MULTIDIM = "wave_multidim"


@dataclass
class ArchitectureGenome:
    """Genotype encoding of an artificial neural organisation."""
    genome_id: str
    topology_family: TopologyFamily
    num_nodes: int
    node_types: List[str]  # e.g. ["sensory", "interneuron", "motor", "modulatory", "workspace"]
    adjacency_list: Dict[int, List[int]]  # src -> [dst1, dst2, ...]
    edge_weights: Dict[str, float]  # "src->dst" -> weight
    communication_mode: CommunicationMode = CommunicationMode.SCALAR
    plasticity_rule: str = "hebbian"  # "hebbian", "stdp", "bcm", "none"
    activation_dynamics: str = "rate"  # "rate", "lif", "kuramoto", "reaction_diffusion"
    neuromodulatory_pathways: Dict[str, List[int]] = field(default_factory=dict)
    hyperedges: List[List[int]] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_hash(self) -> str:
        """Deterministic SHA-256 fingerprint of the architecture genome."""
        data = {
            "family": self.topology_family.value,
            "num_nodes": self.num_nodes,
            "node_types": self.node_types,
            "adj": {str(k): sorted(v) for k, v in self.adjacency_list.items()},
            "weights": {k: round(v, 4) for k, v in sorted(self.edge_weights.items())},
            "comm": self.communication_mode.value,
            "plasticity": self.plasticity_rule,
            "dynamics": self.activation_dynamics,
        }
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def to_matrix(self) -> np.ndarray:
        """Convert adjacency and weights into dense (N, N) matrix."""
        mat = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for src, neighbors in self.adjacency_list.items():
            for dst in neighbors:
                w_key = f"{src}->{dst}"
                mat[src, dst] = self.edge_weights.get(w_key, 1.0)
        return mat

    def mutate(
        self,
        rng: np.random.Generator,
        weight_mutation_rate: float = 0.2,
        structural_mutation_rate: float = 0.1,
    ) -> ArchitectureGenome:
        """Produces a mutated offspring genome with lineage tracking."""
        child_adj = {k: list(v) for k, v in self.adjacency_list.items()}
        child_weights = dict(self.edge_weights)
        child_node_types = list(self.node_types)
        child_num_nodes = self.num_nodes

        # 1. Weight perturbations
        for k in list(child_weights.keys()):
            if rng.random() < weight_mutation_rate:
                child_weights[k] += float(rng.normal(0.0, 0.1))
                child_weights[k] = float(np.clip(child_weights[k], -3.0, 3.0))

        # 2. Structural mutation: add/remove edge
        if rng.random() < structural_mutation_rate:
            src = int(rng.integers(0, child_num_nodes))
            dst = int(rng.integers(0, child_num_nodes))
            if src != dst:
                src_neighbors = child_adj.setdefault(src, [])
                if dst not in src_neighbors:
                    src_neighbors.append(dst)
                    child_weights[f"{src}->{dst}"] = float(rng.normal(0.0, 0.5))
                elif len(src_neighbors) > 1 and rng.random() < 0.5:
                    src_neighbors.remove(dst)
                    child_weights.pop(f"{src}->{dst}", None)

        # 3. Node duplication / neurogenesis mutation
        if rng.random() < (structural_mutation_rate * 0.3) and child_num_nodes < 128:
            new_node = child_num_nodes
            child_num_nodes += 1
            source_template = int(rng.integers(0, child_num_nodes - 1))
            child_node_types.append(child_node_types[source_template])
            child_adj[new_node] = []
            # Inherit subset of connections with noise
            for target in child_adj.get(source_template, []):
                if rng.random() < 0.7:
                    child_adj[new_node].append(target)
                    child_weights[f"{new_node}->{target}"] = float(
                        child_weights.get(f"{source_template}->{target}", 0.5) + rng.normal(0, 0.05)
                    )

        child_id = f"gen_{self.generation + 1}_{rng.integers(10000, 99999)}"
        offspring = ArchitectureGenome(
            genome_id=child_id,
            topology_family=self.topology_family,
            num_nodes=child_num_nodes,
            node_types=child_node_types,
            adjacency_list=child_adj,
            edge_weights=child_weights,
            communication_mode=self.communication_mode,
            plasticity_rule=self.plasticity_rule,
            activation_dynamics=self.activation_dynamics,
            neuromodulatory_pathways=copy.deepcopy(self.neuromodulatory_pathways),
            hyperedges=copy.deepcopy(self.hyperedges),
            generation=self.generation + 1,
            parent_ids=[self.genome_id],
            metadata=copy.deepcopy(self.metadata),
        )
        return offspring

    @classmethod
    def crossover(
        cls,
        parent1: ArchitectureGenome,
        parent2: ArchitectureGenome,
        rng: np.random.Generator,
    ) -> ArchitectureGenome:
        """Recombines two compatible genomes."""
        n_nodes = min(parent1.num_nodes, parent2.num_nodes)
        child_node_types = [
            parent1.node_types[i] if rng.random() < 0.5 else parent2.node_types[i]
            for i in range(n_nodes)
        ]
        child_adj: Dict[int, List[int]] = {}
        child_weights: Dict[str, float] = {}

        for i in range(n_nodes):
            p1_edges = [e for e in parent1.adjacency_list.get(i, []) if e < n_nodes]
            p2_edges = [e for e in parent2.adjacency_list.get(i, []) if e < n_nodes]
            all_edges = list(set(p1_edges + p2_edges))
            selected = [e for e in all_edges if rng.random() < 0.6]
            child_adj[i] = selected
            for dst in selected:
                k = f"{i}->{dst}"
                w1 = parent1.edge_weights.get(k)
                w2 = parent2.edge_weights.get(k)
                if w1 is not None and w2 is not None:
                    child_weights[k] = (w1 + w2) / 2.0
                elif w1 is not None:
                    child_weights[k] = w1
                elif w2 is not None:
                    child_weights[k] = w2
                else:
                    child_weights[k] = float(rng.normal(0, 0.3))

        child_id = f"cross_{max(parent1.generation, parent2.generation) + 1}_{rng.integers(10000, 99999)}"
        return cls(
            genome_id=child_id,
            topology_family=parent1.topology_family if rng.random() < 0.5 else parent2.topology_family,
            num_nodes=n_nodes,
            node_types=child_node_types,
            adjacency_list=child_adj,
            edge_weights=child_weights,
            communication_mode=parent1.communication_mode if rng.random() < 0.5 else parent2.communication_mode,
            plasticity_rule=parent1.plasticity_rule if rng.random() < 0.5 else parent2.plasticity_rule,
            activation_dynamics=parent1.activation_dynamics if rng.random() < 0.5 else parent2.activation_dynamics,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.genome_id, parent2.genome_id],
        )
