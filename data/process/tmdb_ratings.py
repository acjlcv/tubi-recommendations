from data.process.private import TMDB_ACCESS_TOKEN
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

from keybert import KeyBERT
from model2vec import StaticModel


def process_csv(file_path: str):
    df = pd.read_csv(file_path)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    df["genres"] = (
        df["genres"]
        .str.split("·")
        .apply(lambda x: [g.strip() for g in x] if isinstance(x, list) else [])
    )

    df["media_type"] = df["url"].str.extract(r"tubitv\.com/([^/]+)/")

    return df


emb_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
model = KeyBERT(emb_model)


def parse_keywords(s: str):  # mainly to parse from synopsis
    keywords = model.extract_keywords(s)
    return [w for w, p in keywords]  # p for probabilities


headers = {"accept": "application/json", "Authorization": f"Bearer {TMDB_ACCESS_TOKEN}"}


def get_ratings(df):
    res = []
    for i in tqdm(range(df.shape[0])):
        row = df.iloc[i]
        title, year, media_type = row["title"], row["year"], row["media_type"]

        if pd.notna(year):
            params = {"query": title, "year": str(year)}
        else:
            params = {"query": title}

        try:
            api_type = "movie" if media_type == "movies" else "tv"
            endpoint = f"https://api.themoviedb.org/3/search/{api_type}"
            response = requests.get(endpoint, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if data["results"]:
                    movie = data["results"][0]

                    if movie["overview"]:
                        keywords = parse_keywords(movie["overview"])
                        while len(keywords) < 5:
                            keywords.append("")

                        if len(keywords) > 5:
                            keywords = keywords[:5]

                    else:
                        keywords = ["" for _ in range(5)]

                    res.append(
                        {
                            "title": title,
                            "score_avg": movie["vote_average"],
                            "vote_count": movie["vote_count"],
                            "language": movie["original_language"],
                            "popularity": movie["popularity"],
                            "keywords": keywords,
                        }
                    )
                else:
                    res.append(
                        {
                            "title": title,
                            "score_avg": np.nan,
                            "vote_count": np.nan,
                            "language": np.nan,
                            "popularity": np.nan,
                            "keywords": ["" for _ in range(5)],
                        }
                    )
        except Exception as e:
            print(title, e)
            res.append(
                {
                    "title": title,
                    "score_avg": np.nan,
                    "vote_count": np.nan,
                    "language": np.nan,
                    "popularity": np.nan,
                    "keywords": ["" for _ in range(5)],
                }
            )
    return pd.DataFrame(res)


def main(args):
    raw = process_csv(args.file_path)
    rated = get_ratings(raw)

    merged = pd.merge(raw, rated, on="title")

    if args.debug:
        print(merged.head(5))

    if args.write:
        merged.to_csv(args.write_path)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path", dest="file_path", type=str, default="./data/raw/tubi_raw.csv"
    )
    parser.add_argument("-w", "--write", action="store_true")
    parser.add_argument(
        "--write-path",
        dest="write_path",
        type=str,
        default="./data/process/tubi_processed.csv",
    )
    parser.add_argument("-d", dest="debug", action="store_true")

    args = parser.parse_args()
    main(args)
