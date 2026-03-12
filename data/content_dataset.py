import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re

from model2vec import StaticModel

# all genres mentioned in the tubi dataset
GENRES_MAP = {
    "Sci-Fi": 0,
    "Action": 1,
    "Adventure": 2,
    "Drama": 3,
    "Crime": 4,
    "Romance": 5,
    "Foreign/International": 6,
    "War": 7,
    "Thriller": 8,
    "Western": 9,
    "Comedy": 10,
    "Holiday": 11,
    "Kids & Family": 12,
    "Animation": 13,
    "Horror": 14,
    "Fantasy": 15,
    "Musicals": 16,
    "Sport": 17,
    "Adult": 18,
    "Documentary": 19,
    "Mystery": 20,
    "Independent": 21,
    "Music": 22,
    "LGBT": 23,
    "Reality": 24,
    "Lifestyle": 25,
    "Anime": 26,
}

LANGUAGES_MAP = {
    "en": 0,
    "es": 1,
    "de": 2,
    "zh": 3,
    "fr": 4,
    "it": 5,
    "sv": 6,
    "cn": 7,
    "ja": 8,
    "ko": 9,
    "hi": 10,
    "ru": 11,
}


# since dataset is pretty small, im just going to load everything into memory
# fetches static media embedding
class ContentDataset(Dataset):
    def __init__(self, csv_file_path: str = "./data/process/tubi_processed.csv"):
        # fill all empty data
        self.df = pd.read_csv(csv_file_path)
        self.df["score_avg"] = self.df["score_avg"].fillna(0)
        self.df["vote_count"] = self.df["vote_count"].fillna(0)
        self.df["popularity"] = self.df["popularity"].fillna(
            self.df["popularity"].mean()
        )
        self.df["duration"] = self.df["duration"].apply(
            lambda x: self.duration_to_hrs(str(x)) if x else 0.0
        )

        # using 8M for quicker inference + 256 embed size
        self.model = StaticModel.from_pretrained("minishlab/potion-base-8M")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):  # do the embedding of the values here
        row = self.df.iloc[idx]

        # encode into embeddings/ohe\
        keywords = torch.tensor(
            np.array([self.model.encode(k) for k in eval(row["keywords"])]),
            dtype=torch.float32,
        )

        genres = self.encode_genres(eval(row["genres"]))
        language = self.encode_languages(row["language"])

        # ints/floats
        media_type = 1.0 if row["media_type"] == "movies" else 0.0
        score_avg = row["score_avg"]
        vote_count = row["vote_count"]
        year = row["year"]
        duration = row["duration"]
        metrics_vector = torch.tensor(
            [score_avg, vote_count, media_type, duration, year], dtype=torch.float32
        )

        popularity = torch.tensor(row["popularity"], dtype=torch.float32)
        return {
            "keywords": keywords,
            "genres": genres,
            "language": language,
            "popularity": popularity,
            "metrics_vector": metrics_vector,
        }

    # ohe
    def encode_languages(self, lang):
        vec = torch.zeros(len(LANGUAGES_MAP), dtype=torch.float32)
        if lang in LANGUAGES_MAP:
            vec[LANGUAGES_MAP[lang]] = 1.0
        return vec

    # ohe
    def encode_genres(self, genres):
        vec = torch.zeros(len(GENRES_MAP), dtype=torch.float32)
        for g in genres:
            if g in GENRES_MAP:
                vec[GENRES_MAP[g]] = 1.0
        return vec

    def duration_to_hrs(self, duration_str):
        hours_match = re.search(r"(\d+)\s*h", duration_str)
        minutes_match = re.search(r"(\d+)\s*m", duration_str)

        hours = float(hours_match.group(1)) if hours_match else 0.0
        minutes = float(minutes_match.group(1)) if minutes_match else 0.0

        return hours + (minutes / 60.0)

    def get_metadata(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        genres_list = eval(row["genres"]) if isinstance(row["genres"], str) else []
        return {
            "title": row["title"],
            "genres": genres_list,
        }
