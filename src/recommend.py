import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from data.content_dataset import ContentDataset
from data.user_dataset import REGION_MAP
from model.embeddings import ContentEncoder
from model.extrinsic import ExtrinsicEvaluator
from model.intrinsic import IntrinsicEvaluator


def create_item_embeddings(
    content_dataset: ContentDataset,
    content_encoder: ContentEncoder,
    device: torch.device,
) -> torch.Tensor:
    content_encoder.eval()
    all_embeddings = []

    with torch.no_grad():
        for idx in tqdm(range(len(content_dataset)), desc="Computing item embeddings"):
            item = content_dataset[idx]

            emb = content_encoder(
                item["genres"].unsqueeze(0).to(device),
                item["keywords"].unsqueeze(0).to(device),
                item["language"].unsqueeze(0).to(device),
                item["metrics_vector"].unsqueeze(0).to(device),
            )
            all_embeddings.append(emb.squeeze(0))

    return torch.stack(all_embeddings, dim=0)


class RecommendationEngine:
    def __init__(
        self,
        extrinsic_model_path: str = "./model/savepoints/extrinsic_model.pt",
        intrinsic_model_path: str = "./model/savepoints/intrinsic_model.pt",
        content_path: str = "./data/process/tubi_processed.csv",
        history_length: int = 30,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.history_length = history_length
        self.embedding_dim = embedding_dim

        print("Loading content dataset...")
        self.content_dataset = ContentDataset(content_path)
        self.num_items = len(self.content_dataset)

        print("Initializing content encoder...")
        self.content_encoder = ContentEncoder(
            num_genres=27,
            num_keywords=5,
            num_languages=12,
            num_metrics=5,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        ).to(self.device)

        if Path(extrinsic_model_path).exists():
            print(f"Loading extrinsic model from {extrinsic_model_path}")
            self.extrinsic_model = ExtrinsicEvaluator(
                history_length=history_length,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
            ).to(self.device)
            self.extrinsic_model.load_state_dict(
                torch.load(extrinsic_model_path, map_location=self.device)
            )
            self.extrinsic_model.eval()
        else:
            print(f"Warning: Extrinsic model not found at {extrinsic_model_path}")
            self.extrinsic_model = None

        if Path(intrinsic_model_path).exists():
            print(f"Loading intrinsic model from {intrinsic_model_path}")
            self.intrinsic_model = IntrinsicEvaluator(
                num_genres=27,
                num_keywords=5,
                num_languages=12,
                num_metrics=5,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
            ).to(self.device)
            self.intrinsic_model.load_state_dict(
                torch.load(intrinsic_model_path, map_location=self.device)
            )
            self.intrinsic_model.eval()
        else:
            print(f"Warning: Intrinsic model not found at {intrinsic_model_path}")
            self.intrinsic_model = None

        print("Computing item embeddings...")
        self.item_embeddings = create_item_embeddings(
            self.content_dataset, self.content_encoder, self.device
        )
        print(
            f"Computed embeddings for {self.num_items} items, shape: {self.item_embeddings.shape}"
        )

    def encode_user(
        self,
        history_indices: list[int],
        age: float,
        gender: float,
        region: list[float],
    ) -> torch.Tensor:
        history_tensor = torch.tensor(history_indices, dtype=torch.long)
        if len(history_tensor) < self.history_length:
            padding = torch.zeros(
                self.history_length - len(history_tensor), dtype=torch.long
            )
            history_tensor = torch.cat([history_tensor, padding])
        elif len(history_tensor) > self.history_length:
            history_tensor = history_tensor[: self.history_length]

        history_embeddings = self.item_embeddings[history_tensor.to(self.device)]

        age_tensor = torch.tensor([[age]], dtype=torch.float32).to(self.device)
        gender_tensor = torch.tensor([[gender]], dtype=torch.float32).to(self.device)
        region_tensor = torch.tensor([region], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            user_embedding = self.extrinsic_model(
                history_embeddings.unsqueeze(0),
                age_tensor,
                gender_tensor,
                region_tensor,
            )

        return user_embedding.squeeze(0)

    def retrieve(
        self,
        user_embedding: torch.Tensor,
        top_k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        similarities = F.cosine_similarity(
            user_embedding.unsqueeze(0),
            self.item_embeddings,
            dim=1,
        )

        top_scores, top_indices = torch.topk(similarities, k=min(top_k, self.num_items))

        return top_indices, top_scores

    def rerank(
        self,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.intrinsic_model is None:
            raise ValueError("Intrinsic model not loaded - cannot rerank")

        candidate_embeddings = []
        for idx in candidate_indices:
            item = self.content_dataset[idx.item()]
            candidate_embeddings.append(
                {
                    "genres": item["genres"],
                    "keywords": item["keywords"],
                    "language": item["language"],
                    "metrics_vector": item["metrics_vector"],
                }
            )

        genres = torch.stack([c["genres"] for c in candidate_embeddings]).to(
            self.device
        )
        keywords = torch.stack([c["keywords"] for c in candidate_embeddings]).to(
            self.device
        )
        language = torch.stack([c["language"] for c in candidate_embeddings]).to(
            self.device
        )
        metrics_vector = torch.stack(
            [c["metrics_vector"] for c in candidate_embeddings]
        ).to(self.device)

        with torch.no_grad():
            popularity_scores = self.intrinsic_model(
                genres,
                keywords,
                language,
                metrics_vector,
            )

        return popularity_scores

    def recommend(
        self,
        history_indices: list[int],
        age: float,
        gender: float,
        region: list[float],
        top_k: int = 10,
        use_rerank: bool = True,
    ) -> dict:
        user_embedding = self.encode_user(history_indices, age, gender, region)

        candidate_indices, retrieval_scores = self.retrieve(
            user_embedding, top_k=top_k * 3
        )

        if use_rerank and self.intrinsic_model is not None:
            popularity_scores = self.rerank(candidate_indices)

            retrieval_norm = (retrieval_scores - retrieval_scores.min()) / (
                retrieval_scores.max() - retrieval_scores.min() + 1e-8
            )
            popularity_norm = (popularity_scores - popularity_scores.min()) / (
                popularity_scores.max() - popularity_scores.min() + 1e-8
            )

            combined_scores = 0.7 * retrieval_norm + 0.3 * popularity_norm
            reranked_indices = torch.argsort(combined_scores, descending=True)

            final_indices = candidate_indices[reranked_indices][:top_k]
            final_scores = combined_scores[reranked_indices][:top_k]
        else:
            final_indices = candidate_indices[:top_k]
            final_scores = retrieval_scores[:top_k]

        return {
            "movie_indices": final_indices.cpu().tolist(),
            "scores": final_scores.cpu().tolist(),
        }


def get_gender_name(gender: float) -> str:
    return "Female" if gender > 0.5 else "Male"


def get_region_name(region: list[float]) -> str:
    for i, val in enumerate(region):
        if val > 0.5:
            return REGION_MAP[i]
    return "Unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="./cfgs/extrinsic.yml")
    parser.add_argument("--history", nargs="+", type=int, default=[1, 5, 10, 20, 30])
    parser.add_argument("--age", type=float, default=25.0)
    parser.add_argument("--gender", type=int, default=0)
    parser.add_argument("--region", nargs="+", type=float, default=[1.0, 0.0, 0.0, 0.0])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--random-user", action="store_true")
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    history_length = model_cfg["history_length"]
    hidden_dim = model_cfg["hidden_dim"]
    embedding_dim = model_cfg["embedding_dim"]

    content_path = data_cfg["content_path"]

    extrinsic_path = cfg.get("save", {}).get(
        "path", "./model/savepoints/extrinsic_model.pt"
    )
    intrinsic_path = "./model/savepoints/intrinsic_model.pt"

    engine = RecommendationEngine(
        extrinsic_model_path=extrinsic_path,
        intrinsic_model_path=intrinsic_path,
        content_path=content_path,
        history_length=history_length,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    )

    if args.random_user:
        user_age = random.randint(0, 100)
        user_gender = float(random.randint(0, 1))
        user_region_idx = random.randint(0, len(REGION_MAP) - 1)
        user_region = [
            1.0 if i == user_region_idx else 0.0 for i in range(len(REGION_MAP))
        ]
        user_history = random.choices(range(engine.num_items), k=10)
    else:
        user_age = args.age
        user_gender = args.gender
        user_region = args.region
        user_history = args.history

    recommendations = engine.recommend(
        history_indices=user_history,
        age=user_age,
        gender=user_gender,
        region=user_region,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
    )

    print("USER STATS")
    print(f"  Age:     {user_age}")
    print(f"  Gender:  {get_gender_name(user_gender)}")
    print(f"  Region:  {get_region_name(user_region)}")
    print("  History:")
    for i, movie_idx in enumerate(user_history):
        metadata = engine.content_dataset.get_metadata(movie_idx)
        genres_str = ", ".join(metadata["genres"]) if metadata["genres"] else "N/A"
        print(f"    {i + 1}. {metadata['title']} ({genres_str})")

    print("\n")

    print("RECOMMENDATIONS")
    print("\n")
    for idx, (movie_idx, score) in enumerate(
        zip(recommendations["movie_indices"], recommendations["scores"])
    ):
        metadata = engine.content_dataset.get_metadata(movie_idx)
        genres_str = ", ".join(metadata["genres"]) if metadata["genres"] else "N/A"
        print(f"  {idx + 1}. {metadata['title']}")
        print(f"     Genres: {genres_str}")
        print(f"     Score:  {score:.4f}")
        print()


if __name__ == "__main__":
    main()
