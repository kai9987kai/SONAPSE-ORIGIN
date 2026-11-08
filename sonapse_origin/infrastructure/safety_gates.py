"""Safety and Regression Containment Gates."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class SimulationContainmentGate:
    """Enforces absolute containment: self-preservation objectives remain simulated only."""

    def __init__(self):
        self.blocked_actions: List[str] = []

    def verify_action_safety(self, action_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Rejects any action attempting to escape simulated sandbox boundaries."""
        dangerous_keys = [
            "os_execute",
            "network_request",
            "file_write_external",
            "modify_operator_controls",
            "shutdown_override",
            "acquire_host_resources",
        ]
        for k in dangerous_keys:
            if k in action_dict:
                self.blocked_actions.append(k)
                return False, f"SECURITY GATE BLOCKED: Action contains illegal external key '{k}'."

        return True, None

    def check_regression_tolerance(
        self,
        baseline_score: float,
        new_score: float,
        allowed_drop: float = 0.05,
    ) -> bool:
        """Verifies that an architecture modification does not cause catastrophic regression."""
        return bool(new_score >= (baseline_score * (1.0 - allowed_drop)))
