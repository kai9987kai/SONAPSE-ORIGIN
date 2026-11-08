"""Topology Generator for 16 Network Architectures in Structure Forge."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np

from .genome import ArchitectureGenome, CommunicationMode, TopologyFamily


class TopologyGenerator:
    """Generates authentic network graphs across 16 structural topology families."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        family: TopologyFamily,
        num_nodes: int = 32,
        comm_mode: CommunicationMode = CommunicationMode.SCALAR,
        **kwargs: Any,
    ) -> ArchitectureGenome:
        """Dispatcher for generating a genome of the requested topology family."""
        method_name = f"_gen_{family.value}"
        generator_fn = getattr(self, method_name, self._gen_recurrent)
        return generator_fn(num_nodes=num_nodes, comm_mode=comm_mode, **kwargs)

    # 1. Feedforward Directed Graphs
    def _gen_feedforward(
        self, num_nodes: int, comm_mode: CommunicationMode, layers: int = 4, **kwargs: Any
    ) -> ArchitectureGenome:
        nodes_per_layer = max(1, num_nodes // layers)
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}
        node_types: List[str] = []

        layer_assignments = [min(layers - 1, i // nodes_per_layer) for i in range(num_nodes)]
        for i, l in enumerate(layer_assignments):
            if l == 0:
                node_types.append("sensory")
            elif l == layers - 1:
                node_types.append("motor")
            else:
                node_types.append("interneuron")

        for i in range(num_nodes):
            l_i = layer_assignments[i]
            for j in range(num_nodes):
                l_j = layer_assignments[j]
                if l_j == l_i + 1:
                    if self.rng.random() < 0.6:
                        adj[i].append(j)
                        weights[f"{i}->{j}"] = float(self.rng.normal(0.5, 0.2))

        return ArchitectureGenome(
            genome_id=f"ff_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.FEEDFORWARD,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 2. Recurrent Directed Graphs
    def _gen_recurrent(
        self, num_nodes: int, comm_mode: CommunicationMode, density: float = 0.2, **kwargs: Any
    ) -> ArchitectureGenome:
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}
        node_types = ["recurrent_node"] * num_nodes
        node_types[0] = "sensory"
        node_types[-1] = "motor"

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and self.rng.random() < density:
                    adj[i].append(j)
                    # Dale's law approximation: 80% excitatory, 20% inhibitory
                    sign = 1.0 if self.rng.random() < 0.8 else -1.0
                    weights[f"{i}->{j}"] = float(sign * abs(self.rng.normal(0.4, 0.15)))

        return ArchitectureGenome(
            genome_id=f"rec_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.RECURRENT,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 3. Small-World Networks (Watts-Strogatz)
    def _gen_small_world(
        self, num_nodes: int, comm_mode: CommunicationMode, k: int = 4, p: float = 0.15, **kwargs: Any
    ) -> ArchitectureGenome:
        k = min(num_nodes - 1, max(2, k))
        G = nx.watts_strogatz_graph(num_nodes, k, p, seed=int(self.rng.integers(0, 100000)))
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for u, v in G.edges():
            adj[u].append(v)
            adj[v].append(u)
            weights[f"{u}->{v}"] = float(self.rng.normal(0.5, 0.2))
            weights[f"{v}->{u}"] = float(self.rng.normal(0.5, 0.2))

        return ArchitectureGenome(
            genome_id=f"sw_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.SMALL_WORLD,
            num_nodes=num_nodes,
            node_types=["small_world_node"] * num_nodes,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 4. Scale-Free Networks (Barabási-Albert)
    def _gen_scale_free(
        self, num_nodes: int, comm_mode: CommunicationMode, m: int = 2, **kwargs: Any
    ) -> ArchitectureGenome:
        m = min(num_nodes - 1, max(1, m))
        G = nx.barabasi_albert_graph(num_nodes, m, seed=int(self.rng.integers(0, 100000)))
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        # Classify high-degree nodes as hubs
        degrees = dict(G.degree())
        hub_threshold = np.quantile(list(degrees.values()), 0.85)
        node_types = ["hub" if degrees[i] >= hub_threshold else "peripheral" for i in range(num_nodes)]

        for u, v in G.edges():
            adj[u].append(v)
            adj[v].append(u)
            weights[f"{u}->{v}"] = float(self.rng.normal(0.5, 0.15))
            weights[f"{v}->{u}"] = float(self.rng.normal(0.5, 0.15))

        return ArchitectureGenome(
            genome_id=f"sf_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.SCALE_FREE,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 5. Hierarchical Modular Networks
    def _gen_hierarchical_modular(
        self, num_nodes: int, comm_mode: CommunicationMode, modules: int = 4, **kwargs: Any
    ) -> ArchitectureGenome:
        modules = max(2, min(modules, num_nodes // 2))
        node_module = [i % modules for i in range(num_nodes)]
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                # Higher intra-module probability (0.5), lower inter-module probability (0.05)
                p = 0.5 if node_module[i] == node_module[j] else 0.05
                if self.rng.random() < p:
                    adj[i].append(j)
                    w = self.rng.normal(0.6 if node_module[i] == node_module[j] else 0.2, 0.1)
                    weights[f"{i}->{j}"] = float(w)

        return ArchitectureGenome(
            genome_id=f"hm_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.HIERARCHICAL_MODULAR,
            num_nodes=num_nodes,
            node_types=[f"module_{node_module[i]}" for i in range(num_nodes)],
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 6. Rich-Club Networks
    def _gen_rich_club(
        self, num_nodes: int, comm_mode: CommunicationMode, club_fraction: float = 0.2, **kwargs: Any
    ) -> ArchitectureGenome:
        club_size = max(2, int(num_nodes * club_fraction))
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}
        node_types = ["rich_club_core" if i < club_size else "feeder" for i in range(num_nodes)]

        # Dense connectivity between rich-club members
        for i in range(club_size):
            for j in range(club_size):
                if i != j and self.rng.random() < 0.85:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.8, 0.1))

        # Feeder connections to core
        for i in range(club_size, num_nodes):
            targets = self.rng.choice(club_size, size=min(2, club_size), replace=False)
            for t in targets:
                adj[i].append(int(t))
                adj[int(t)].append(i)
                weights[f"{i}->{t}"] = float(self.rng.normal(0.4, 0.1))
                weights[f"{t}->{i}"] = float(self.rng.normal(0.4, 0.1))

        return ArchitectureGenome(
            genome_id=f"rc_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.RICH_CLUB,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 7. Multilayer Connectomes (sensory, associative, motor, regulatory)
    def _gen_multilayer_connectome(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        quarters = num_nodes // 4
        layers = ["sensory", "associative", "motor", "regulatory"]
        node_types = []
        for i in range(num_nodes):
            layer_idx = min(3, i // quarters) if quarters > 0 else 0
            node_types.append(layers[layer_idx])

        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            t_i = node_types[i]
            for j in range(num_nodes):
                if i == j:
                    continue
                t_j = node_types[j]
                # Intra-layer or specific inter-layer bridges
                if t_i == t_j and self.rng.random() < 0.25:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.4, 0.1))
                elif t_i == "sensory" and t_j == "associative" and self.rng.random() < 0.3:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.5, 0.1))
                elif t_i == "associative" and t_j == "motor" and self.rng.random() < 0.3:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.5, 0.1))
                elif t_i == "regulatory" and self.rng.random() < 0.15:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.3, 0.05))

        return ArchitectureGenome(
            genome_id=f"ml_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.MULTILAYER_CONNECTOME,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 8. Dynamic Hypergraphs (multi-node hyperedges)
    def _gen_dynamic_hypergraph(
        self, num_nodes: int, comm_mode: CommunicationMode, num_hyperedges: int = 6, **kwargs: Any
    ) -> ArchitectureGenome:
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}
        hyperedges: List[List[int]] = []

        # Create multi-node hyperedges (order 3 to 5)
        for _ in range(num_hyperedges):
            order = int(self.rng.integers(3, min(6, num_nodes + 1)))
            nodes = self.rng.choice(num_nodes, size=order, replace=False).tolist()
            hyperedges.append(nodes)
            # Expand hyperedge into pairwise projection for standard adjacency fallback
            for u in nodes:
                for v in nodes:
                    if u != v and v not in adj[u]:
                        adj[u].append(v)
                        weights[f"{u}->{v}"] = float(self.rng.normal(0.3, 0.1))

        return ArchitectureGenome(
            genome_id=f"hg_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.DYNAMIC_HYPERGRAPH,
            num_nodes=num_nodes,
            node_types=["hyper_node"] * num_nodes,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
            hyperedges=hyperedges,
        )

    # 9. Spatial Neural Fields (geometric 2D layout)
    def _gen_spatial_neural_field(
        self, num_nodes: int, comm_mode: CommunicationMode, length_scale: float = 0.25, **kwargs: Any
    ) -> ArchitectureGenome:
        coords = self.rng.uniform(0.0, 1.0, size=(num_nodes, 2))
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    dist = float(np.linalg.norm(coords[i] - coords[j]))
                    # Mexican-hat distance-dependent connection: local excitation, surround inhibition
                    prob = np.exp(-dist / length_scale)
                    if self.rng.random() < prob:
                        adj[i].append(j)
                        w = 1.0 * np.exp(-dist**2 / (2 * 0.15**2)) - 0.4 * np.exp(-dist**2 / (2 * 0.4**2))
                        weights[f"{i}->{j}"] = float(w)

        return ArchitectureGenome(
            genome_id=f"snf_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.SPATIAL_NEURAL_FIELD,
            num_nodes=num_nodes,
            node_types=["field_unit"] * num_nodes,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
            metadata={"coordinates": coords.tolist()},
        )

    # 10. Graphs with Evolving Communities
    def _gen_evolving_community(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        community = [0 if i < num_nodes // 2 else 1 for i in range(num_nodes)]
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    p = 0.4 if community[i] == community[j] else 0.08
                    if self.rng.random() < p:
                        adj[i].append(j)
                        weights[f"{i}->{j}"] = float(self.rng.normal(0.5, 0.15))

        return ArchitectureGenome(
            genome_id=f"ec_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.EVOLVING_COMMUNITY,
            num_nodes=num_nodes,
            node_types=[f"community_{community[i]}" for i in range(num_nodes)],
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 11. Coupled Oscillator Populations
    def _gen_coupled_oscillator(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}
        # Ring plus random shortcuts (small world for phase synchronization)
        for i in range(num_nodes):
            left = (i - 1) % num_nodes
            right = (i + 1) % num_nodes
            adj[i].extend([left, right])
            weights[f"{i}->{left}"] = 0.5
            weights[f"{i}->{right}"] = 0.5
            if self.rng.random() < 0.2:
                shortcut = int(self.rng.integers(0, num_nodes))
                if shortcut != i and shortcut not in adj[i]:
                    adj[i].append(shortcut)
                    weights[f"{i}->{shortcut}"] = 0.3

        return ArchitectureGenome(
            genome_id=f"osc_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.COUPLED_OSCILLATOR,
            num_nodes=num_nodes,
            node_types=["kuramoto_oscillator"] * num_nodes,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=CommunicationMode.PHASE_CODED,
            activation_dynamics="kuramoto",
        )

    # 12. Hybrid Neural-Symbolic Structures
    def _gen_hybrid_neural_symbolic(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        symbolic_count = max(2, num_nodes // 4)
        node_types = ["neural_worker"] * (num_nodes - symbolic_count) + ["symbolic_concept"] * symbolic_count
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if node_types[i] == "symbolic_concept" and node_types[j] == "symbolic_concept":
                    # Logic / relational binding
                    if self.rng.random() < 0.4:
                        adj[i].append(j)
                        weights[f"{i}->{j}"] = 1.0  # Discrete relation
                elif node_types[i] == "neural_worker" and node_types[j] == "symbolic_concept":
                    # Neural predicate grounding
                    if self.rng.random() < 0.25:
                        adj[i].append(j)
                        weights[f"{i}->{j}"] = float(self.rng.normal(0.4, 0.1))
                elif self.rng.random() < 0.1:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.3, 0.1))

        return ArchitectureGenome(
            genome_id=f"ns_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.HYBRID_NEURAL_SYMBOLIC,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 13. Networks with Shared Global Workspaces
    def _gen_global_workspace(
        self, num_nodes: int, comm_mode: CommunicationMode, workspace_size: int = 4, **kwargs: Any
    ) -> ArchitectureGenome:
        workspace_size = max(1, min(workspace_size, num_nodes // 3))
        node_types = ["specialist_module"] * (num_nodes - workspace_size) + ["global_workspace_hub"] * workspace_size
        workspace_indices = list(range(num_nodes - workspace_size, num_nodes))
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        # Dense interconnect among workspace hub nodes
        for w1 in workspace_indices:
            for w2 in workspace_indices:
                if w1 != w2:
                    adj[w1].append(w2)
                    weights[f"{w1}->{w2}"] = 0.9

        # Bidirectional connectivity between specialist modules and the workspace
        for i in range(num_nodes - workspace_size):
            # Connect to at least one workspace node for ignition/broadcast
            target_hub = int(self.rng.choice(workspace_indices))
            adj[i].append(target_hub)
            adj[target_hub].append(i)
            weights[f"{i}->{target_hub}"] = 0.7  # Ingestion
            weights[f"{target_hub}->{i}"] = 0.7  # Global broadcast

        return ArchitectureGenome(
            genome_id=f"gw_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.GLOBAL_WORKSPACE,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
            metadata={"workspace_indices": workspace_indices},
        )

    # 14. Networks with Local Workspaces Only
    def _gen_local_workspace_only(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        # Multiple isolated clusters with no centralized global hub
        cluster_size = max(4, num_nodes // 3)
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            c_i = i // cluster_size
            for j in range(num_nodes):
                c_j = j // cluster_size
                if i != j and c_i == c_j and self.rng.random() < 0.6:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.6, 0.1))

        return ArchitectureGenome(
            genome_id=f"lw_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.LOCAL_WORKSPACE_ONLY,
            num_nodes=num_nodes,
            node_types=["local_cluster_node"] * num_nodes,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 15. Systems with Competing or Cooperating Regulatory Centres
    def _gen_competing_regulatory(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        # Two regulatory centers: appetitive center (0) and defensive/aversive center (1)
        node_types = ["regulatory_appetitive", "regulatory_aversive"] + ["worker"] * (num_nodes - 2)
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        # Mutual inhibition between the two regulatory centers
        adj[0].append(1)
        adj[1].append(0)
        weights["0->1"] = -0.8
        weights["1->0"] = -0.8

        # Each controls different downstream populations
        for i in range(2, num_nodes):
            if i % 2 == 0:
                adj[0].append(i)
                weights[f"0->{i}"] = 0.5
            else:
                adj[1].append(i)
                weights[f"1->{i}"] = 0.5
            # Feedback to regulatory centers
            if self.rng.random() < 0.3:
                target = int(self.rng.choice([0, 1]))
                adj[i].append(target)
                weights[f"{i}->{target}"] = 0.3

        return ArchitectureGenome(
            genome_id=f"cr_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.COMPETING_REGULATORY,
            num_nodes=num_nodes,
            node_types=node_types,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
        )

    # 16. Networks with Dynamic Structural Rewiring
    def _gen_dynamic_rewiring(
        self, num_nodes: int, comm_mode: CommunicationMode, **kwargs: Any
    ) -> ArchitectureGenome:
        # Starts with sparse random graph; configured for high structural plasticity
        adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        weights: Dict[str, float] = {}

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and self.rng.random() < 0.15:
                    adj[i].append(j)
                    weights[f"{i}->{j}"] = float(self.rng.normal(0.3, 0.1))

        return ArchitectureGenome(
            genome_id=f"dr_{self.rng.integers(1000, 9999)}",
            topology_family=TopologyFamily.DYNAMIC_REPEATED_REWIRING,
            num_nodes=num_nodes,
            node_types=["plastic_unit"] * num_nodes,
            adjacency_list=adj,
            edge_weights=weights,
            communication_mode=comm_mode,
            plasticity_rule="structural_hebbian",
            metadata={"rewiring_frequency": 5, "pruning_threshold": 0.05},
        )
