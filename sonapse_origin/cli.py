"""Command-Line Interface for SONAPSE-ORIGIN."""
from __future__ import annotations

import argparse
import sys
import uvicorn

from .experiments import (
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
    run_experiment_k,
    run_experiment_l,
    run_experiment_m,
    run_experiment_n,
    run_experiment_o,
    run_experiment_p,
)
from .infrastructure.database import ExperimentDatabase
from .adapters.sonapse_adapter import SonapseLabAdapter
from .adapters.poseidon_adapter import PoseidonAdapter


def cmd_run_experiments(args: argparse.Namespace) -> int:
    print("=" * 80)
    print("SONAPSE-ORIGIN // EXTENDED EXPERIMENTAL PROGRAMME (A - P)")
    print("=" * 80)

    all_experiments = [
        ("A", run_experiment_a),
        ("B", run_experiment_b),
        ("C", run_experiment_c),
        ("D", run_experiment_d),
        ("E", run_experiment_e),
        ("F", run_experiment_f),
        ("G", run_experiment_g),
        ("H", run_experiment_h),
        ("I", run_experiment_i),
        ("J", run_experiment_j),
        ("K", run_experiment_k),
        ("L", run_experiment_l),
        ("M", run_experiment_m),
        ("N", run_experiment_n),
        ("O", run_experiment_o),
        ("P", run_experiment_p),
    ]

    selected = all_experiments
    if args.suite == "initial":
        selected = all_experiments[:10]
    elif args.suite == "frontier":
        selected = all_experiments[10:]

    results = []
    for code, exp_fn in selected:
        print(f"\n[RUNNING] Experiment {code}...")
        res = exp_fn(sample_size=args.sample_size)
        results.append(res)
        print(f"  Title: {res['title']}")
        print(f"  Theory: {res['theory']} | Metric: {res['target_metric']}")
        print(f"  Treatment: {res['treatment_mean']:.4f} | Control: {res['control_mean']:.4f}")
        print(f"  Effect Size (Cohen's d): {res['effect_size_cohens_d']:.3f} | p-value: {res['p_value']:.4e}")
        print(f"  Status: {res['status'].upper()} | Receipt: {res['receipt_hash'][:16]}...")

    print("\n" + "=" * 80)
    print("EXPERIMENTAL PROGRAMME SUMMARY:")
    supported_count = sum(1 for r in results if r["status"] == "supported")
    print(f"Total Evaluated: {len(results)} | Supported: {supported_count} | Falsified: {len(results) - supported_count}")
    print("=" * 80)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    print(f"Starting SONAPSE-ORIGIN Scientific Observatory on http://{args.host}:{args.port}")
    uvicorn.run("sonapse_origin.observatory.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print("=" * 60)
    print("SONAPSE-ORIGIN // PLATFORM STATUS")
    print("=" * 60)
    db = ExperimentDatabase()
    with db._get_connection() as conn:
        n_genomes = conn.execute("SELECT COUNT(*) FROM architecture_genomes").fetchone()[0]
        n_hyp = conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        n_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        n_disc = conn.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0]

    print(f"Architecture Genomes Recorded: {n_genomes}")
    print(f"Preregistered Hypotheses:     {n_hyp}")
    print(f"Executed Experiments:          {n_exp}")
    print(f"Candidate Discoveries:         {n_disc}")

    sonapse_ad = SonapseLabAdapter()
    poseidon_ad = PoseidonAdapter()
    print(f"Sonapse-LAB Connected:         {sonapse_ad.is_available}")
    print(f"Poseidon Connected:            {poseidon_ad.is_available}")
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SONAPSE-ORIGIN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-experiments
    p_run = subparsers.add_parser("run-experiments", help="Run experimental suite (A-P)")
    p_run.add_argument("--sample-size", type=int, default=15, help="Sample size per condition")
    p_run.add_argument("--suite", type=str, choices=["all", "initial", "frontier"], default="all", help="Subset to execute")
    p_run.set_defaults(func=cmd_run_experiments)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch live Scientific Observatory web server")
    p_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_serve.add_argument("--port", type=int, default=8000, help="Port")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    p_serve.set_defaults(func=cmd_serve)

    # status
    p_status = subparsers.add_parser("status", help="Display platform status and database metrics")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
