import torch
import torch.nn as nn

from model.embeddings import UserEncoder

EMBEDDING_DIM = 256


class ExtrinsicEvaluator(nn.Module):  # two towers
    def __init__(
        self, history_length: int = 30, hidden_dim: int = 256, embedding_dim: int = 256
    ):
        super().__init__()
        self.user_encoder = UserEncoder(
            history_length=history_length,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.user_encoder(history_embeddings, age, gender, region)
        return self.output_mlp(user_emb)
