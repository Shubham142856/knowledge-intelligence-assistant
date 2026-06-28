import torch
import numpy as np

class VYORSurpriseGate:
    def __init__(self, alpha=0.1, delta=1.0):
        """
        Surprise Gate initialize karega adaptive threshold tracking ke sath.
        delta=1.0 ensures Huber Loss behaves quadratically for small errors 
        and linearly for high surprise outliers.
        """
        self.delta = delta
        self.alpha = alpha  # Momentum weight for adaptive updates if needed
        self.rolling_history = []  # Isme saare pichle chunks ke losses save honge
        self.warmup_chunks = 10    # Minimum itne chunks chahiye percentile active karne ke liye

    def compute_huber_loss(self, current_vector: torch.Tensor, memory_weights: torch.Tensor) -> float:
        """
        Calculates mathematical Huber Loss between incoming chunk vector and memory state.
        Formula: 
        If |error| <= delta: 0.5 * error^2
        Else: delta * (|error| - 0.5 * delta)
        """
        # Element-wise absolute error distance
        error = torch.abs(current_vector - memory_weights)
        
        # Huber Loss Logic
        quadratic = torch.clamp(error, max=self.delta)
        linear = error - quadratic
        
        loss_tensor = 0.5 * (quadratic ** 2) + self.delta * linear
        mean_loss = torch.mean(loss_tensor).item()
        
        return mean_loss

    def update_and_route(self, current_loss: float) -> tuple[str, float]:
        """
        Decides whether to route to Titans LTM or Partner's Qdrant Vector DB 
        based on the dynamic 80th percentile threshold.
        """
        # History mein current loss ko append karo
        self.rolling_history.append(current_loss)
        
        # Warmup phase check: Jab tak 10 chunks na ho jayein, use safe fixed threshold
        if len(self.rolling_history) < self.warmup_chunks:
            # Baseline threshold during warmup
            threshold = 0.35
            decision = "memory_updated" if current_loss >= threshold else "save_to_qdrant"
            return decision, threshold
        
        # Actual Adaptive Logic: Calculate 80th Percentile of history
        adaptive_threshold = float(np.percentile(self.rolling_history, 80))
        
        # Route logic
        if current_loss >= adaptive_threshold:
            decision = "memory_updated"  # Dynamic High Surprise!
        else:
            decision = "save_to_qdrant"   # Low Surprise, standard storage
            
        return decision, adaptive_threshold