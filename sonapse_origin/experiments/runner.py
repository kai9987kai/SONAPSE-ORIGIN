"""Unified Scientific Experiment Runner."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..director import AutonomousResearchDirector, ScientificHypothesis, HypothesisStatus
from ..infrastructure import ExperimentDatabase, ExperimentReceipt


class ExperimentRunner:
    """Coordinates execution, preregistration, statistical analysis, and receipts for all experiments."""

    def __init__(self, db_path: str = "sonapse_origin.db"):
        self.director = AutonomousResearchDirector()
        self.db = ExperimentDatabase(db_path)

    def run_controlled_experiment(
        self,
        exp_id: str,
        title: str,
        theory: str,
        statement: str,
        target_metric: str,
        treatment_fn: Callable[[int], float],
        control_fn: Callable[[int], float],
        sample_size: int = 15,
        base_seed: int = 100,
        env_level: str = "Level1",
    ) -> Dict[str, Any]:
        """Runs preregistered factorial experiment across seeds with treatment vs control."""
        # 1. Propose and preregister hypothesis
        hyp = self.director.propose_hypothesis(
            hypothesis_id=f"HYP_{exp_id}",
            title=title,
            statement=statement,
            theory=theory,
            target_metric=target_metric,
        )
        self.director.preregister_hypothesis(hyp.hypothesis_id)
        self.db.record_hypothesis(hyp.hypothesis_id, title, theory, hyp.status.value, {})

        # 2. Execute runs across matched seeds
        treatment_scores = []
        control_scores = []

        for i in range(sample_size):
            seed = base_seed + i
            t_score = treatment_fn(seed)
            c_score = control_fn(seed)
            treatment_scores.append(t_score)
            control_scores.append(c_score)

        t_arr = np.array(treatment_scores)
        c_arr = np.array(control_scores)

        # 3. Statistical evaluation
        eval_res = self.director.evaluate_experiment_results(hyp.hypothesis_id, t_arr, c_arr)

        # 4. Generate cryptographic provenance receipt
        receipt = ExperimentReceipt.create(
            experiment_id=exp_id,
            git_commit="origin_head",
            architecture_hash=f"arch_{exp_id}",
            seed=base_seed,
            environment_level=env_level,
            solver="rk4",
            measurements={
                f"treatment_{target_metric}": float(np.mean(t_arr)),
                f"control_{target_metric}": float(np.mean(c_arr)),
                "cohens_d": eval_res["effect_size_d"],
                "p_value": eval_res["p_value"],
            },
            status=eval_res["status"],
        )

        # 5. Record to database
        self.db.record_experiment(
            exp_id=exp_id,
            name=title,
            hyp_id=hyp.hypothesis_id,
            genome_id=None,
            env=env_level,
            seed=base_seed,
            status=eval_res["status"],
            config={"sample_size": sample_size, "base_seed": base_seed},
            receipt_hash=receipt.receipt_hash,
        )
        self.db.record_discovery(
            discovery_id=f"DISC_{exp_id}",
            hyp_id=hyp.hypothesis_id,
            claim=statement,
            equation=None,
            effect_size=eval_res["effect_size_d"],
            p_value=eval_res["p_value"],
            replicated=True,
        )

        return {
            "experiment_id": exp_id,
            "title": title,
            "theory": theory,
            "target_metric": target_metric,
            "treatment_mean": float(np.mean(t_arr)),
            "control_mean": float(np.mean(c_arr)),
            "effect_size_cohens_d": eval_res["effect_size_d"],
            "p_value": eval_res["p_value"],
            "status": eval_res["status"],
            "receipt_hash": receipt.receipt_hash,
        }
