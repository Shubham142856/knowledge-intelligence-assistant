"""
surprise_gate.py — VYORSurpriseGate (Adaptive Surprise-Gated Routing)

Implements:
1. Huber Loss error evaluation between input and memory states.
2. Momentum-based surprise accumulator (MIRAS framework).
3. Dynamic 80th-percentile thresholding.
4. Adaptive forgetting parameter (alpha_t) estimation.
"""

import math
import torch
import numpy as np

class VYORSurpriseGate:
    def __init__(self, alpha=0.1, delta=1.0, gamma=0.9, percentile=80):
        """
        Initialises Surprise Gate with momentum tracking.
        
        Args:
            alpha: Learning/adaptation rate weight.
            delta: Huber Loss transition threshold.
            gamma: Momentum decay weight for the surprise accumulator (0.9 standard).
            percentile: Surprise percentile threshold (60 to 95, default 80).
        """
        self.delta = delta
        self.alpha = alpha
        self.gamma = gamma
        self.percentile = percentile
        
        # Momentum surprise tracker
        self.surprise_accumulator = 0.0
        self.last_alpha_t = 1.0
        
        self.rolling_history = []  # Tracks accumulated surprise values
        self.warmup_chunks = 10

    def compute_huber_loss(self, current_vector: torch.Tensor, memory_weights: torch.Tensor) -> float:
        """
        Computes robust Huber Loss between incoming vector and neural memory recall reference.
        """
        # Ensure dimensions match
        v1 = current_vector.view(-1)
        v2 = memory_weights.view(-1)
        
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
        
        # Absolute error calculation
        error = torch.abs(v1 - v2)
        
        quadratic = torch.clamp(error, max=self.delta)
        linear = error - quadratic
        
        loss_tensor = 0.5 * (quadratic ** 2) + self.delta * linear
        return float(torch.mean(loss_tensor).item())

    def update_and_route(self, current_loss: float) -> tuple[str, float]:
        """
        Updates the momentum surprise accumulator and determines routing destination.
        
        Returns:
            decision (str) : "memory_updated" (Titans LTM) or "save_to_qdrant" (Qdrant)
            threshold (float): Active percentile threshold
        """
        # 1. Update the momentum-based surprise accumulator (MIRAS style)
        self.surprise_accumulator = (self.gamma * self.surprise_accumulator) + ((1.0 - self.gamma) * current_loss)
        self.rolling_history.append(self.surprise_accumulator)
        
        # 2. Extract baseline dynamic threshold
        if len(self.rolling_history) < self.warmup_chunks:
            # Safe initial boundary during warmup
            threshold = 0.35
            decision = "memory_updated" if self.surprise_accumulator >= threshold else "save_to_qdrant"
            self.last_alpha_t = 1.0 if decision == "memory_updated" else 0.0
            return decision, threshold
            
        # Calculate adaptive percentile of historical accumulated surprise
        adaptive_threshold = float(np.percentile(self.rolling_history, self.percentile))
        
        # 3. Dynamic forgetting weight (alpha_t) via sigmoid activation scaled around difference
        diff = self.surprise_accumulator - adaptive_threshold
        self.last_alpha_t = 1.0 / (1.0 + math.exp(-5.0 * diff)) # steep sigmoid
        
        # 4. Route decision
        if self.surprise_accumulator >= adaptive_threshold:
            decision = "memory_updated"
        else:
            decision = "save_to_qdrant"
            
        return decision, adaptive_threshold