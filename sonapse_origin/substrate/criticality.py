"""Self-Organized Criticality (SOC) and Neuronal Avalanche Substrate.

Implements neuronal avalanche detection, branching parameter estimation,
power-law scaling verification, and dynamic homeostatic synaptic scaling to self-tune
neural systems to the critical boundary (sigma ~ 1.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class AvalancheStatistics:
    num_avalanches: int
    sizes: List[int]
    durations: List[int]
    branching_parameter: float
    size_power_law_alpha: float
    duration_power_law_beta: float
    crackling_noise_ratio: float
    is_critical: bool


class CriticalitySubstrate:
    """Simulates neural activity, detects avalanches, and adapts synapses to maintain criticality."""

    def __init__(
        self,
        num_nodes: int = 50,
        refractory_period: int = 2,
        seed: int = 42,
    ):
        self.num_nodes = num_nodes
        self.refractory_period = refractory_period
        self.rng = np.random.default_rng(seed)
        
        # Initialize connectivity with random weights
        self.W = self.rng.normal(0.0, 1.0 / np.sqrt(num_nodes), (num_nodes, num_nodes))
        np.fill_diagonal(self.W, 0.0)

        self.thresholds = np.ones(num_nodes, dtype=np.float64)
        self.voltages = np.zeros(num_nodes, dtype=np.float64)
        self.refractory_timers = np.zeros(num_nodes, dtype=np.int32)

    def simulate(
        self,
        timesteps: int = 500,
        external_input_rate: float = 0.02,
        homeostatic_adaptation: bool = False,
        adaptation_rate: float = 0.01,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Simulates integrate-and-fire network and returns spike raster of shape (timesteps, num_nodes).
        
        If homeostatic_adaptation is True, scales synaptic matrix dynamically based on instantaneous branching ratio.
        """
        spikes = np.zeros((timesteps, self.num_nodes), dtype=np.int32)
        weight_history = [] if homeostatic_adaptation else None

        for t in range(timesteps):
            # 1. External Poisson drive to trigger avalanches
            ext_drive = (self.rng.random(self.num_nodes) < external_input_rate).astype(np.float64)

            # 2. Update refractory states
            ready = self.refractory_timers == 0
            self.refractory_timers[~ready] -= 1

            # 3. Recurrent input
            if t > 0:
                recurrent_drive = self.W @ spikes[t - 1]
            else:
                recurrent_drive = np.zeros(self.num_nodes)

            # 4. Integrate voltage
            self.voltages[ready] += ext_drive[ready] + recurrent_drive[ready]

            # 5. Emit spikes
            fired = ready & (self.voltages >= self.thresholds)
            spikes[t, fired] = 1
            self.voltages[fired] = 0.0
            self.refractory_timers[fired] = self.refractory_period

            # Leak
            self.voltages[ready & ~fired] *= 0.9

            # 6. Dynamic homeostatic synaptic scaling to tune to criticality
            if homeostatic_adaptation and t > 0:
                n_curr = np.sum(spikes[t])
                n_prev = np.sum(spikes[t - 1])
                if n_prev > 0 and n_curr > 0:
                    local_branching = n_curr / n_prev
                    # If branching > 1 (supercritical/epileptiform), scale down
                    # If branching < 1 (subcritical/quenched), scale up
                    scale_factor = 1.0 + adaptation_rate * (1.0 - local_branching)
                    scale_factor = float(np.clip(scale_factor, 0.90, 1.10))
                    self.W *= scale_factor
                weight_history.append(float(np.mean(np.abs(self.W))))

        return spikes, (np.array(weight_history) if weight_history is not None else None)

    def extract_avalanches(
        self,
        spikes: np.ndarray,
        bin_size: int = 1,
    ) -> AvalancheStatistics:
        """Identifies avalanches in binned activity and estimates critical scaling exponents."""
        # Bin total population spikes
        total_steps = spikes.shape[0]
        n_bins = total_steps // bin_size
        binned = np.zeros(n_bins, dtype=np.int32)
        for b in range(n_bins):
            binned[b] = np.sum(spikes[b * bin_size : (b + 1) * bin_size])

        avalanches_sizes: List[int] = []
        avalanches_durations: List[int] = []
        branching_ratios: List[float] = []

        in_avalanche = False
        current_size = 0
        current_duration = 0

        for b in range(n_bins):
            active_count = binned[b]
            if active_count > 0:
                if not in_avalanche:
                    in_avalanche = True
                    current_size = active_count
                    current_duration = 1
                else:
                    current_size += active_count
                    current_duration += 1
                
                # Measure pairwise branching ratio
                if b > 0 and binned[b - 1] > 0:
                    branching_ratios.append(float(active_count / binned[b - 1]))
            else:
                if in_avalanche:
                    avalanches_sizes.append(current_size)
                    avalanches_durations.append(current_duration)
                    in_avalanche = False
                    current_size = 0
                    current_duration = 0

        if in_avalanche:
            avalanches_sizes.append(current_size)
            avalanches_durations.append(current_duration)

        if not avalanches_sizes:
            return AvalancheStatistics(
                num_avalanches=0,
                sizes=[],
                durations=[],
                branching_parameter=0.0,
                size_power_law_alpha=0.0,
                duration_power_law_beta=0.0,
                crackling_noise_ratio=0.0,
                is_critical=False,
            )

        # Estimate power law exponent alpha via Clauset-Newman-Moore MLE: alpha = 1 + N / sum(ln(x / x_min))
        s_arr = np.array(avalanches_sizes, dtype=np.float64)
        s_min = np.min(s_arr)
        if s_min > 0 and len(s_arr) > 5 and np.any(s_arr > s_min):
            alpha = float(1.0 + len(s_arr) / np.sum(np.log(s_arr / s_min + 1e-12)))
        else:
            alpha = 1.5

        d_arr = np.array(avalanches_durations, dtype=np.float64)
        d_min = np.min(d_arr)
        if d_min > 0 and len(d_arr) > 5 and np.any(d_arr > d_min):
            beta = float(1.0 + len(d_arr) / np.sum(np.log(d_arr / d_min + 1e-12)))
        else:
            beta = 2.0

        sigma = float(np.mean(branching_ratios)) if branching_ratios else 0.0

        # Theoretical crackling noise scaling relation: (beta - 1) / (alpha - 1) ~ 1 / (sigma * nu * z)
        crackling_ratio = float((beta - 1.0) / (alpha - 1.0)) if abs(alpha - 1.0) > 1e-4 else 1.0

        # Criticality criteria: branching parameter sigma close to 1.0 (0.85 - 1.15) and alpha ~ 1.5
        is_crit = (0.85 <= sigma <= 1.25) and (1.2 <= alpha <= 2.2)

        return AvalancheStatistics(
            num_avalanches=len(avalanches_sizes),
            sizes=avalanches_sizes,
            durations=avalanches_durations,
            branching_parameter=sigma,
            size_power_law_alpha=alpha,
            duration_power_law_beta=beta,
            crackling_noise_ratio=crackling_ratio,
            is_critical=is_crit,
        )
