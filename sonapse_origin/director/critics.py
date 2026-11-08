"""Adversarial Scientific Critics: Methodologist, Skeptic, Statistician, and Replication."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


class StatisticianCritic:
    """Computes rigorous statistical validation: effect size (Cohen's d), p-values, 95% CIs."""

    def analyze(
        self,
        treatment_samples: np.ndarray,
        control_samples: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates Welch's t-test, Cohen's d, and confidence intervals."""
        n_t = len(treatment_samples)
        n_c = len(control_samples)
        mean_t = float(np.mean(treatment_samples))
        mean_c = float(np.mean(control_samples))
        var_t = float(np.var(treatment_samples, ddof=1)) if n_t > 1 else 1e-4
        var_c = float(np.var(control_samples, ddof=1)) if n_c > 1 else 1e-4

        diff = mean_t - mean_c

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(treatment_samples, control_samples, equal_var=False)

        # Pooled standard deviation for Cohen's d
        s_pooled = np.sqrt(((n_t - 1) * var_t + (n_c - 1) * var_c) / max(1, n_t + n_c - 2))
        cohens_d = float(diff / (s_pooled + 1e-8))

        # Standard error of the difference
        se_diff = np.sqrt(var_t / max(1, n_t) + var_c / max(1, n_c))
        ci_lower = float(diff - 1.96 * se_diff)
        ci_upper = float(diff + 1.96 * se_diff)

        return {
            "mean_treatment": mean_t,
            "mean_control": mean_c,
            "mean_difference": diff,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": cohens_d,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "statistically_significant": bool(p_val < 0.05),
        }


class SkepticCritic:
    """Detects anthropomorphism, unwarranted claims of sentience, and confounding variables."""

    BANNED_SENTIENCE_PHRASES = [
        "is conscious",
        "feels pain",
        "has feelings",
        "proves sentience",
        "experienced emotion",
        "artificial soul",
        "subjective experience achieved",
    ]

    def audit_claim(self, claim_text: str) -> Dict[str, Any]:
        """Flags unwarranted claims of subjective experience or consciousness."""
        flags = []
        lower = claim_text.lower()
        for phrase in self.BANNED_SENTIENCE_PHRASES:
            if phrase in lower:
                flags.append(f"Unwarranted phenomenal claim detected: '{phrase}'")

        approved = len(flags) == 0
        return {
            "approved": approved,
            "flags": flags,
            "recommendation": (
                "Approved: objective operational phrasing."
                if approved
                else "REVISE: rephrase in terms of observable computational dynamics and theory-specific proxies."
            ),
        }


class MethodologistCritic:
    """Validates experimental design, parameter-matched controls, and intervention hygiene."""

    def review_experiment_plan(
        self,
        treatment_capacity: int,
        control_capacity: int,
        sample_size: int,
    ) -> Dict[str, Any]:
        """Verifies capacity matching and statistical power."""
        issues = []
        if abs(treatment_capacity - control_capacity) > (0.1 * treatment_capacity):
            issues.append(
                f"Model capacity mismatch: treatment has {treatment_capacity} params, control has {control_capacity}"
            )
        if sample_size < 10:
            issues.append(f"Sample size {sample_size} is underpowered (minimum recommended: 10)")

        valid = len(issues) == 0
        return {
            "valid_design": valid,
            "issues": issues,
        }
