"""Cryptographic Provenance Receipts and Audit Certificates."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentReceipt:
    experiment_id: str
    timestamp: float
    platform_version: str
    git_commit: str
    architecture_hash: str
    seed: int
    environment_level: str
    solver: str
    measurements_summary: Dict[str, float]
    falsification_status: str
    receipt_hash: str = ""

    def calculate_receipt_hash(self) -> str:
        """Deterministic cryptographic signature of the entire experiment run."""
        payload = {
            "exp_id": self.experiment_id,
            "git": self.git_commit,
            "arch_hash": self.architecture_hash,
            "seed": self.seed,
            "env": self.environment_level,
            "solver": self.solver,
            "metrics": {k: round(v, 5) for k, v in sorted(self.measurements_summary.items())},
            "status": self.falsification_status,
        }
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    @classmethod
    def create(
        cls,
        experiment_id: str,
        git_commit: str,
        architecture_hash: str,
        seed: int,
        environment_level: str,
        solver: str,
        measurements: Dict[str, float],
        status: str,
        platform_version: str = "0.1.0",
    ) -> ExperimentReceipt:
        receipt = cls(
            experiment_id=experiment_id,
            timestamp=time.time(),
            platform_version=platform_version,
            git_commit=git_commit,
            architecture_hash=architecture_hash,
            seed=seed,
            environment_level=environment_level,
            solver=solver,
            measurements_summary=measurements,
            falsification_status=status,
        )
        receipt.receipt_hash = receipt.calculate_receipt_hash()
        return receipt
