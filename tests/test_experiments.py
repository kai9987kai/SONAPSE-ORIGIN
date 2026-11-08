import pytest
from sonapse_origin.experiments import (
    run_experiment_a,
    run_experiment_b,
    run_experiment_c,
    run_experiment_d,
    run_experiment_e,
    run_experiment_f,
    run_experiment_g,
    run_experiment_h,
    run_experiment_i,
    run_experiment_j,
)


def test_experiments_a_through_j_execution():
    # Run each experiment with a small sample size (e.g. 5) to verify flawless execution & receipts
    experiments = [
        run_experiment_a,
        run_experiment_b,
        run_experiment_c,
        run_experiment_d,
        run_experiment_e,
        run_experiment_f,
        run_experiment_g,
        run_experiment_h,
        run_experiment_i,
        run_experiment_j,
    ]

    for exp_fn in experiments:
        res = exp_fn(sample_size=5)
        assert "experiment_id" in res
        assert "status" in res
        assert "effect_size_cohens_d" in res
        assert "receipt_hash" in res
        assert len(res["receipt_hash"]) == 64
