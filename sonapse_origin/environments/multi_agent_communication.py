"""Multi-Agent Emergent Communication & Dialect Formation Environment.

Models collective communication emergence:
- Agents share an environment with hidden ecological states / hazards
- Agents communicate over a discrete symbolic channel with alphabet {c_1, ..., c_K}
- Measures communicative success, mutual information I(Symbol; Hazard), and linguistic convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DialectMetrics:
    mutual_information_symbol_hazard: float
    communicative_accuracy: float
    lexicon_entropy: float
    consensus_score: float


class EmergentCommunicationWorld:
    """Simulates communicative games between sender and receiver agents under viability pressures."""

    def __init__(
        self,
        num_hazards: int = 4,
        alphabet_size: int = 6,
        seed: int = 42,
    ):
        self.num_hazards = num_hazards
        self.alphabet_size = alphabet_size
        self.rng = np.random.default_rng(seed)

        # Sender policy: mapping from hazard (0..num_hazards-1) to logits over alphabet
        self.sender_weights = self.rng.normal(0, 1.0, (num_hazards, alphabet_size))
        # Receiver policy: mapping from symbol to logits over predicted hazards
        self.receiver_weights = self.rng.normal(0, 1.0, (alphabet_size, num_hazards))

    @staticmethod
    def _softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
        exp_l = np.exp((logits - np.max(logits)) / max(0.01, temp))
        return exp_l / np.sum(exp_l)

    def train_dialects(
        self,
        episodes: int = 400,
        learning_rate: float = 0.05,
        temperature: float = 0.8,
    ) -> DialectMetrics:
        """Trains sender and receiver through reinforcement / policy gradient to form a shared code."""
        cooccurrence = np.zeros((self.num_hazards, self.alphabet_size), dtype=np.float64)
        correct_predictions = 0

        for ep in range(episodes):
            hazard = int(self.rng.integers(0, self.num_hazards))
            # Sender emits symbol
            sender_probs = self._softmax(self.sender_weights[hazard], temp=temperature)
            symbol = int(self.rng.choice(self.alphabet_size, p=sender_probs))

            # Receiver decodes hazard
            receiver_probs = self._softmax(self.receiver_weights[symbol], temp=temperature)
            predicted_hazard = int(self.rng.choice(self.num_hazards, p=receiver_probs))

            reward = 1.0 if predicted_hazard == hazard else -0.5
            if reward > 0:
                correct_predictions += 1

            # Policy gradient update (REINFORCE)
            # Sender grad
            s_grad = -sender_probs
            s_grad[symbol] += 1.0
            self.sender_weights[hazard] += learning_rate * reward * s_grad

            # Receiver grad
            r_grad = -receiver_probs
            r_grad[predicted_hazard] += 1.0
            self.receiver_weights[symbol] += learning_rate * reward * r_grad

        # Post-training evaluation phase to evaluate the converged dialect without exploration noise
        eval_queries = 150
        eval_correct = 0
        for _ in range(eval_queries):
            hazard = int(self.rng.integers(0, self.num_hazards))
            sender_probs = self._softmax(self.sender_weights[hazard], temp=0.2)
            symbol = int(np.argmax(sender_probs))
            receiver_probs = self._softmax(self.receiver_weights[symbol], temp=0.2)
            predicted_hazard = int(np.argmax(receiver_probs))

            if predicted_hazard == hazard:
                eval_correct += 1
            cooccurrence[hazard, symbol] += 1.0

        # Calculate mutual information between hazard and emitted symbols
        p_hs = cooccurrence / np.sum(cooccurrence)
        p_h = np.sum(p_hs, axis=1)
        p_s = np.sum(p_hs, axis=0)

        mi = 0.0
        for h in range(self.num_hazards):
            for s in range(self.alphabet_size):
                if p_hs[h, s] > 1e-12:
                    mi += p_hs[h, s] * np.log2(p_hs[h, s] / (p_h[h] * p_s[s] + 1e-12))

        # Lexicon entropy
        p_s_clean = p_s[p_s > 1e-12]
        lex_entropy = float(-np.sum(p_s_clean * np.log2(p_s_clean)))
        accuracy = float(correct_predictions / episodes)

        # Consensus score: average certainty of sender mapping
        row_maxes = np.max(cooccurrence / (np.sum(cooccurrence, axis=1, keepdims=True) + 1e-8), axis=1)
        consensus = float(np.mean(row_maxes))

        return DialectMetrics(
            mutual_information_symbol_hazard=float(max(0.0, mi)),
            communicative_accuracy=accuracy,
            lexicon_entropy=lex_entropy,
            consensus_score=consensus,
        )
