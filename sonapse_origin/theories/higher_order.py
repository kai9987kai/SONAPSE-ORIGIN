"""Higher-Order Representation (HOT) and Metacognitive Sensitivity Experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score


class HigherOrderRepresentationAssay:
    """Evaluates metacognitive self-modelling, confidence calibration, and Type-2 SDT."""

    def __init__(self, first_order_dim: int = 16, meta_dim: int = 8, seed: int = 42):
        self.first_order_dim = first_order_dim
        self.meta_dim = meta_dim
        self.rng = np.random.default_rng(seed)

        # First-order task classifier: input -> class prediction
        self.W_task = self.rng.normal(0, 0.4, (2, first_order_dim))
        # Second-order metacognitive network: internal hidden state -> confidence in correctness
        self.W_meta = self.rng.normal(0, 0.3, (1, first_order_dim))

    def evaluate_metacognitive_sensitivity(
        self,
        num_trials: int = 300,
        noise_level: float = 0.5,
    ) -> Dict[str, float]:
        """Runs perceptual discrimination trials and tests confidence calibration & Type-2 ROC."""
        # True binary classes: 0 or 1
        true_labels = self.rng.integers(0, 2, num_trials)
        # First-order representations: true class signal + noise
        signals = np.zeros((num_trials, self.first_order_dim))
        for i in range(num_trials):
            target_val = 1.0 if true_labels[i] == 1 else -1.0
            signals[i] = target_val * 0.4 + self.rng.normal(0, noise_level, self.first_order_dim)

        # First-order task decision: argmax(W_task @ x)
        logits = signals @ self.W_task.T  # (num_trials, 2)
        decisions = np.argmax(logits, axis=1)
        # Correctness: 1 if decision == true_label else 0
        correct = (decisions == true_labels).astype(int)
        accuracy = float(np.mean(correct))

        # Second-order confidence prediction: sigmoid(W_meta @ x)
        conf_logits = (signals @ self.W_meta.T).flatten()
        # Train meta-weights briefly via logistic regression gradient step to model self-accuracy
        for _ in range(20):
            preds = 1.0 / (1.0 + np.exp(-conf_logits))
            grad = (preds - correct) @ signals
            self.W_meta -= 0.01 * grad[np.newaxis, :]
            conf_logits = (signals @ self.W_meta.T).flatten()

        confidence = 1.0 / (1.0 + np.exp(-conf_logits))

        # Type-2 ROC: Ability of confidence to discriminate correct from incorrect trials
        if len(np.unique(correct)) > 1:
            type2_auc = float(roc_auc_score(correct, confidence))
        else:
            type2_auc = 0.5

        # Expected Calibration Error (ECE)
        bins = np.linspace(0.0, 1.0, 6)
        bin_accuracies = []
        bin_confidences = []
        bin_weights = []
        for b_low, b_high in zip(bins[:-1], bins[1:]):
            in_bin = (confidence >= b_low) & (confidence < b_high)
            if np.any(in_bin):
                bin_accuracies.append(np.mean(correct[in_bin]))
                bin_confidences.append(np.mean(confidence[in_bin]))
                bin_weights.append(np.sum(in_bin) / num_trials)

        ece = float(np.sum(np.array(bin_weights) * np.abs(np.array(bin_accuracies) - np.array(bin_confidences))))

        return {
            "first_order_accuracy": accuracy,
            "mean_confidence": float(np.mean(confidence)),
            "type2_roc_auc": type2_auc,
            "expected_calibration_error": ece,
        }
