"""Initial & Advanced Frontier Experimental Programme: Experiments A through V."""

from .runner import ExperimentRunner
from .exp_a_architecture_integration import run_experiment_a
from .exp_b_homeostasis_learning import run_experiment_b
from .exp_c_global_workspace import run_experiment_c
from .exp_d_oscillatory_communication import run_experiment_d
from .exp_e_structural_plasticity import run_experiment_e
from .exp_f_sleep_replay import run_experiment_f
from .exp_g_metacognitive_self_modelling import run_experiment_g
from .exp_h_open_ended_evolution import run_experiment_h
from .exp_i_affective_regulation import run_experiment_i
from .exp_j_candidate_law_discovery import run_experiment_j
from .exp_k_allostatic_active_inference import run_experiment_k
from .exp_l_phase_amplitude_coupling import run_experiment_l
from .exp_m_causal_emergence import run_experiment_m
from .exp_n_pci_lempel_ziv import run_experiment_n
from .exp_o_hypergraph_synergy import run_experiment_o
from .exp_p_metamorphic_rewiring import run_experiment_p
from .exp_q_self_organized_criticality import run_experiment_q
from .exp_r_precision_predictive_coding import run_experiment_r
from .exp_s_partial_information_decomposition import run_experiment_s
from .exp_t_iit_mechanism_phi import run_experiment_t
from .exp_u_bioelectric_morphogenesis import run_experiment_u
from .exp_v_multi_agent_communication import run_experiment_v

__all__ = [
    "ExperimentRunner",
    "run_experiment_a",
    "run_experiment_b",
    "run_experiment_c",
    "run_experiment_d",
    "run_experiment_e",
    "run_experiment_f",
    "run_experiment_g",
    "run_experiment_h",
    "run_experiment_i",
    "run_experiment_j",
    "run_experiment_k",
    "run_experiment_l",
    "run_experiment_m",
    "run_experiment_n",
    "run_experiment_o",
    "run_experiment_p",
    "run_experiment_q",
    "run_experiment_r",
    "run_experiment_s",
    "run_experiment_t",
    "run_experiment_u",
    "run_experiment_v",
]
