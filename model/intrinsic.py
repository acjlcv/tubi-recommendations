import torch
import torch.nn as nn

from model.embeddings import ContentEncoder

EMBEDDING_DIM = 256

class IntrinsicEvaluator(nn.Module):
    def __init__(self, num_genres:int = 27, num_languages:int = 12, num_keywords:int = 5, num_metrics:int = 5, hidden_dim: int = 256, embedding_dim:int = 256):
        super().__init__()
        self.content_encoder = ContentEncoder(
            num_genres=num_genres, num_keywords=num_keywords, num_languages=num_languages, num_metrics=num_metrics, hidden_dim=hidden_dim, output_dim=embedding_dim
        )
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        genre_vector: torch.Tensor,
        keyword_embeddings: torch.Tensor,
        language_vector: torch.Tensor,
        metrics_vector: torch.Tensor,
    ) -> torch.Tensor:
        content_emb = self.content_encoder(
            genre_vector, keyword_embeddings, language_vector, metrics_vector
        )
        popularity = self.predictor(content_emb)
        return popularity.squeeze(-1) #predict the objective popularity of the movie
