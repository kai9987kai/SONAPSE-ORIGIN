"""SQLite Experiment Database and Audit Ledger."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ExperimentDatabase:
    """Persistent SQLite database tracking architectures, experiments, and measurements."""

    def __init__(self, db_path: str = "sonapse_origin.db"):
        self.db_path = db_path
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            cur = conn.cursor()
            # 1. architectures & genomes
            cur.execute("""
                CREATE TABLE IF NOT EXISTS architecture_genomes (
                    genome_id TEXT PRIMARY KEY,
                    topology_family TEXT NOT NULL,
                    num_nodes INTEGER NOT NULL,
                    genome_hash TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    data_json TEXT NOT NULL
                )
            """)
            # 2. hypotheses
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    theory TEXT NOT NULL,
                    status TEXT NOT NULL,
                    preregistered_at REAL NOT NULL,
                    data_json TEXT NOT NULL
                )
            """)
            # 3. experiments
            cur.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    hypothesis_id TEXT,
                    genome_id TEXT,
                    environment TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    config_json TEXT NOT NULL,
                    receipt_hash TEXT
                )
            """)
            # 4. measurements
            cur.execute("""
                CREATE TABLE IF NOT EXISTS measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)
            # 5. discoveries
            cur.execute("""
                CREATE TABLE IF NOT EXISTS discoveries (
                    discovery_id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    equation TEXT,
                    effect_size REAL,
                    p_value REAL,
                    replicated INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
                )
            """)
            # 6. audit receipts
            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    receipt_hash TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    data_json TEXT NOT NULL
                )
            """)
            conn.commit()

    def record_genome(self, genome_id: str, family: str, num_nodes: int, g_hash: str, gen: int, data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO architecture_genomes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (genome_id, family, num_nodes, g_hash, gen, time.time(), json.dumps(data)),
            )

    def record_hypothesis(self, hyp_id: str, title: str, theory: str, status: str, data: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO hypotheses VALUES (?, ?, ?, ?, ?, ?)",
                (hyp_id, title, theory, status, time.time(), json.dumps(data)),
            )

    def record_experiment(
        self,
        exp_id: str,
        name: str,
        hyp_id: Optional[str],
        genome_id: Optional[str],
        env: str,
        seed: int,
        status: str,
        config: Dict[str, Any],
        receipt_hash: Optional[str] = None,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (exp_id, name, hyp_id, genome_id, env, seed, status, time.time(), time.time(), json.dumps(config), receipt_hash),
            )

    def log_measurement(self, exp_id: str, step: int, metric_name: str, metric_value: float) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO measurements (experiment_id, step, metric_name, metric_value) VALUES (?, ?, ?, ?)",
                (exp_id, step, metric_name, metric_value),
            )

    def record_discovery(
        self,
        discovery_id: str,
        hyp_id: str,
        claim: str,
        equation: Optional[str],
        effect_size: float,
        p_value: float,
        replicated: bool,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO discoveries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (discovery_id, hyp_id, claim, equation, effect_size, p_value, 1 if replicated else 0, time.time()),
            )
