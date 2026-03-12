import torch
import torch.nn as nn


# two tower architecture, encode diff value embeddings
class ContentEncoder(nn.Module):
    def __init__(
        self,
        num_genres: int = 27,
        num_keywords = 5,
        num_languages:int = 12,
        num_metrics: int = 5,
        keyword_dim: int = 256,
        hidden_dim: int = 256,
        output_dim = 256,
    ):  # num genres derived from the num in the dataset
        super().__init__()
        self.keyword_dim = num_keywords * keyword_dim #minish-lab/potoin-base-8M == 256

        self.genre_fc = nn.Linear(num_genres, hidden_dim)
        self.keyword_fc = nn.Linear(self.keyword_dim, hidden_dim)
        self.language_fc = nn.Linear(num_languages, hidden_dim // 4)
        self.metrics_fc = nn.Linear(num_metrics, hidden_dim // 2)

        self.output_fc = nn.Sequential(
            nn.Linear(
                hidden_dim + hidden_dim + hidden_dim // 4 + hidden_dim // 2, hidden_dim
            ),  # conncat diff emb into one large emb
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        genre_vector: torch.Tensor,
        keyword_embeddings: torch.Tensor,
        language_vector: torch.Tensor,
        metrics_vector: torch.Tensor, #size of 4 score_avg, vote_count, media_type, year
    ) -> torch.Tensor:
        genre_emb = self.genre_fc(genre_vector)


        keyword_emb = self.keyword_fc(
            keyword_embeddings.view(keyword_embeddings.size(0), -1)
        )
        language_emb = self.language_fc(language_vector)
        metrics_emb = self.metrics_fc(metrics_vector)

        combined = torch.cat([genre_emb, keyword_emb, language_emb, metrics_emb], dim=1)
        output = self.output_fc(combined)
        return output

class UserEncoder(nn.Module):
    def __init__(
        self, history_length: int = 30, hidden_dim: int = 256, output_dim: int = 256
    ):
        super().__init__()
        self.history_fc = nn.Linear(history_length * output_dim, hidden_dim)

        self.age_fc = nn.Linear(1, 64)
        self.gender_fc = nn.Linear(1, 32)
        self.region_fc = nn.Linear(4, 32)

        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 32 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        region: torch.Tensor,
    ) -> torch.Tensor:
        history_emb = self.history_fc(
            history_embeddings.view(history_embeddings.size(0), -1)
        )

        age_emb = self.age_fc(age)
        gender_emb = self.gender_fc(gender)
        region_emb = self.region_fc(region)

        combined = torch.cat([history_emb, age_emb, gender_emb, region_emb], dim=1)
        output = self.output_fc(combined)
        return output
