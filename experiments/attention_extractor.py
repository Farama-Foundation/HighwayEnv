import math
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KinematicAttentionExtractor(BaseFeaturesExtractor):
    """
    Expects observations shaped either:
      - (batch, vehicles_count, features_dim)  e.g. (B, 5, 5)
      - or (batch, vehicles_count*features_dim) (flattened), which we reshape.
    Row 0 = ego, rows 1.. = neighbour "slots".
    """

    def __init__(self, observation_space, features_dim: int = 128, d_model: int = 32):
        super().__init__(observation_space, features_dim)

        assert len(observation_space.shape) in (2, 1), f"Unexpected obs shape: {observation_space.shape}"

        if len(observation_space.shape) == 2:
            self.vehicles_count, self.feat_dim = observation_space.shape
        else:
            # Flattened case: vehicles_count * feat_dim
            # If use a flat obs space -> set manually
            raise ValueError("Flattened observation_space not supported in this extractor.")

        self.n_nei = self.vehicles_count - 1
        if self.n_nei <= 0:
            raise ValueError("vehicles_count must be >= 2 for attention over neighbours.")

        # Embed each vehicle feature vector -> d_model
        self.embed = nn.Linear(self.feat_dim, d_model)

        # Single-head scaled dot-product attention: q from ego, k/v from neighbours
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # Final projection to SB3 features_dim
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        self.last_attention = None  # (batch, n_nei)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations: (B, V, F)
        if observations.dim() == 2:
            # If feed flattened obs by mistake, this will fail loudly
            raise ValueError(f"Expected 3D obs (B,V,F), got {observations.shape}")

        x = observations
        ego = x[:, 0, :]           # (B, F)
        nei = x[:, 1:, :]          # (B, N, F)

        ego_e = self.embed(ego)    # (B, D)
        nei_e = self.embed(nei)    # (B, N, D)

        q = self.to_q(ego_e)       # (B, D)        # what the ego car is looking for
        k = self.to_k(nei_e)       # (B, N, D)     # what the neighbours offer
        v = self.to_v(nei_e)       # (B, N, D)

        # Scores: (B, N)
        scores = torch.einsum("bd,bnd->bn", q, k) / math.sqrt(k.shape[-1])   # how relevant each neighbour is to the ego right now

        # Mask using presence flag if features include it at index 0:
        # presence == 1 means slot is occupied, 0 means empty
        presence = nei[:, :, 0]  # (B, N)
        mask = presence > 0.5

        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=1)  # (B, N)
        # If all neighbours are masked, softmax can produce NaNs -> fallback to uniform
        if self.n_nei > 0:
            attn = torch.where(
                torch.isnan(attn),
                torch.full_like(attn, 1.0 / self.n_nei),
                attn,
            )

        # Context: (B, D)
        context = torch.einsum("bn,bnd->bd", attn, v)

        self.last_attention = attn.detach()  # store for evaluation/visualisation

        # Combine ego embedding + neighbour context
        features = torch.cat([ego_e, context], dim=1)  # (B, 2D)
        return self.out(features)
