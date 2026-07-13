"""
src/core/titans_memory_core.py — Google Titans & MIRAS Neural Memory Core

Implements:
1. Neural LTM (Multi-Layer Perceptron mapped via test-time gradient updates)
2. STM (Sliding-window self-attention)
3. Learnable Persistent Tokens
4. Three architectural variants: MAC, MAG, MAL configurable at runtime.
5. Automatic hardware device allocation (CPU & CUDA GPU).
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger("vyor_ai.titans_core")


class NeuralLTM(nn.Module):
    """
    Neural Long-Term Memory (LTM) represented as an associative feed-forward network.
    It maps key embeddings to value representations via gradient updates at test time.
    """
    def __init__(self, dim: int, hidden_dim: int = 512):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        # Huber Loss for robust reconstruction
        self.loss_fn = nn.HuberLoss(delta=1.0)
        self.optimizer = optim.SGD(self.mlp.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recall/retrieve value representation for key `x`.
        x shape: (batch, seq_len, dim) or (batch, dim)
        """
        return self.mlp(x)

    def update_memory(self, x: torch.Tensor, alpha_t: float = 0.0, max_grad_norm: float = 1.0) -> float:
        """
        Performs a test-time gradient descent update on LTM weights.
        Attempts to minimize the reconstruction loss: L = HuberLoss(MLP(x), x).
        Applies adaptive weight decay (forgetting) proportional to alpha_t.
        """
        self.mlp.train()
        self.optimizer.zero_grad()
        
        # Apply forgetting/decay to weights based on alpha_t before update
        if alpha_t > 0.0:
            with torch.no_grad():
                for param in self.mlp.parameters():
                    param.data.mul_(1.0 - 0.01 * alpha_t)
        
        # Reconstruction target is the input itself (auto-associative memory)
        reconstruction = self.forward(x)
        loss = self.loss_fn(reconstruction, x)
        
        loss.backward()
        
        # Gradient clipping to prevent NaN/explosions
        nn.utils.clip_grad_norm_(self.mlp.parameters(), max_grad_norm)
        
        self.optimizer.step()
        self.mlp.eval()
        return loss.item()


class ShortTermMemory(nn.Module):
    """
    Short-Term Memory (STM) utilizing sliding-window self-attention.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention over the sequence.
        x shape: (batch, seq_len, dim)
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


class TitansMemoryCore(nn.Module):
    """
    The unified Titans Memory Core combining LTM, STM, and Persistent Tokens.
    Supports MAC, MAG, and MAL variants.
    """
    def __init__(
        self, 
        dim: int = 384, 
        variant: str = "MAC", 
        num_persistent_tokens: int = 8,
        chunk_size: int = 64
    ):
        super().__init__()
        self.dim = dim
        self.variant = variant.upper().strip()
        self.chunk_size = chunk_size
        
        # Detect CUDA / CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Titans Core: Using device allocation -> {self.device}")

        # Components
        self.ltm = NeuralLTM(dim).to(self.device)
        self.stm = ShortTermMemory(dim).to(self.device)
        
        # Learnable Persistent Tokens
        self.num_persistent_tokens = num_persistent_tokens
        if num_persistent_tokens > 0:
            self.persistent_tokens = nn.Parameter(
                torch.randn(1, num_persistent_tokens, dim)
            )
        else:
            self.persistent_tokens = None
            
        # Gating parameter for MAG
        if self.variant == "MAG":
            self.gate_layer = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            ).to(self.device)

        self.to(self.device)
        log.info(f"Titans Memory Core initialised (variant={self.variant}, dim={dim})")

    def learn_new_info(self, retrieved_vector: torch.Tensor, alpha_t: float = 0.0) -> float:
        """
        Performs test-time gradient update on the LTM.
        """
        # Ensure tensor is on the correct device
        x = retrieved_vector.to(self.device)
        
        # NaN / Inf validation guards
        if not torch.isfinite(x).all():
            log.warning("Titans Core: input vector contains NaN/Inf. Skipping LTM update.")
            return 0.0
            
        try:
            with torch.enable_grad():
                loss = self.ltm.update_memory(x, alpha_t=alpha_t)
            log.info(f"Titans Core LTM: Memory updated at test-time (decay alpha={alpha_t:.4f}). Reconstruction Loss = {loss:.4f}")
            return loss
        except Exception as e:
            log.error(f"Titans Core LTM: Failed to update memory: {e}")
            return 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute forward pass based on active variant.
        x shape: (batch, seq_len, dim)
        """
        x = x.to(self.device)
        batch_size, seq_len, _ = x.shape

        # Inject Persistent Tokens if configured
        if self.persistent_tokens is not None:
            # Broadcast persistent tokens to batch size
            p_tokens = self.persistent_tokens.expand(batch_size, -1, -1)
            x = torch.cat([p_tokens, x], dim=1)

        if self.variant == "MAC":
            return self._forward_mac(x)
        elif self.variant == "MAG":
            return self._forward_mag(x)
        elif self.variant == "MAL":
            return self._forward_mal(x)
        else:
            log.warning(f"Unknown variant '{self.variant}', falling back to MAC.")
            return self._forward_mac(x)

    def _forward_mac(self, x: torch.Tensor) -> torch.Tensor:
        """Memory as Context: LTM context tokens prepended to STM input."""
        # Retrieve LTM context vectors for the inputs
        with torch.no_grad():
            ltm_context = self.ltm(x)
            
        # Concatenate LTM context to sequence dimensions
        fused = torch.cat([ltm_context, x], dim=1)
        
        # Process combined sequence through STM self-attention
        stm_out = self.stm(fused)
        
        # Return only the tokens corresponding to the original inputs
        original_tokens_start = ltm_context.size(1)
        return stm_out[:, original_tokens_start:, :]

    def _forward_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Memory as Gate: LTM gating weights applied to STM outputs."""
        # Retrieve LTM states
        with torch.no_grad():
            ltm_state = self.ltm(x)
            
        # Process STM self-attention
        stm_state = self.stm(x)
        
        # Compute sigmoid gating factor based on LTM & STM values
        concat_state = torch.cat([ltm_state, stm_state], dim=-1)
        gate = self.gate_layer(concat_state)
        
        # Gate the attention output
        return gate * stm_state + (1.0 - gate) * ltm_state

    def _forward_mal(self, x: torch.Tensor) -> torch.Tensor:
        """Memory as Layer: LTM and STM stacked sequentially."""
        # Retrieve LTM state
        with torch.no_grad():
            ltm_state = self.ltm(x)
            
        # Pass LTM output as sequential input into STM
        return self.stm(ltm_state + x)
