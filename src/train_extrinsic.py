import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from data.content_dataset import ContentDataset
from data.user_dataset import UserDataset
from model.embeddings import ContentEncoder
from model.extrinsic import ExtrinsicEvaluator

def create_item_embeddings(
    content_dataset: ContentDataset,
    content_encoder: ContentEncoder,
    device: torch.device,
) -> torch.Tensor:
    content_encoder.eval()
    all_embeddings = []

    with torch.no_grad():
        for idx in range(len(content_dataset)):
            item = content_dataset[idx]

            emb = content_encoder(
                item["genres"].unsqueeze(0).to(device),
                item["keywords"].unsqueeze(0).to(device),
                item["language"].unsqueeze(0).to(device),
                item["metrics_vector"].unsqueeze(0).to(device),
            )
            all_embeddings.append(emb.squeeze(0))

    return torch.stack(all_embeddings, dim=0)


def get_history_embeddings(
    indices: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    l = item_embeddings[indices]
    return l


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="./cfgs/extrinsic.yml")
    args = parser.parse_args()

    with open(args.cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    save_cfg = cfg["save"]

    batch_size = training_cfg["batch_size"]
    epochs = training_cfg["epochs"]
    learning_rate = training_cfg["learning_rate"]
    train_ratio = training_cfg["train_ratio"]
    val_ratio = training_cfg["val_ratio"]

    history_length = model_cfg["history_length"]
    hidden_dim = model_cfg["hidden_dim"]
    embedding_dim = model_cfg["embedding_dim"]
    user_size = model_cfg["user_size"]

    content_path = data_cfg["content_path"]

    save_path = save_cfg["path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading content dataset...")
    content_dataset = ContentDataset(content_path)
    num_items = len(content_dataset)

    print("Initializing content encoder and computing item embeddings...")
    content_encoder = ContentEncoder(
        num_genres=27,
        num_keywords=5,
        num_languages=12,
        num_metrics=5,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
    ).to(device)

    ckpt = torch.load("./model/savepoints/intrinsic_model.pt", map_location=device)
    content_encoder.load_state_dict(
        ckpt, strict=False
    )  # since content encoder is trained inside intrinsic evaluator

    item_embeddings = create_item_embeddings(content_dataset, content_encoder, device)
    print(f"Computed embeddings for {num_items} items, shape: {item_embeddings.shape}")

    print("Creating user dataset...")
    user_dataset = UserDataset(size=user_size, history_length=history_length)

    print("Creating data loaders...")
    total_users = len(user_dataset)
    train_size = int(train_ratio * total_users)
    val_size = int(val_ratio * total_users)
    test_size = total_users - train_size - val_size

    train_users, val_users, test_users = random_split(
        user_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_users, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_users, batch_size=batch_size)
    test_loader = DataLoader(test_users, batch_size=batch_size)

    print("Creating Extrinsic Model...")
    model = ExtrinsicEvaluator(
        history_length=history_length,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = Adam(
        list(model.parameters()), lr=learning_rate
    )
    criterion = MSELoss()

    train_losses = []
    val_losses = []

    print("Starting training...")
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            history_indices = batch["watch_history"].to(device)
            age = batch["age"].to(device)
            gender = batch["gender"].to(device)
            region = batch["region"].to(device)

            history_embeddings = get_history_embeddings(
                history_indices, item_embeddings
            )

            target_embeddings = history_embeddings.mean(dim=1)

            optimizer.zero_grad()

            user_embeddings = model(
                history_embeddings,
                age,
                gender,
                region,
            )

            loss = criterion(user_embeddings, target_embeddings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                history_indices = batch["watch_history"].to(device)
                age = batch["age"].to(device)
                gender = batch["gender"].to(device)
                region = batch["region"].to(device)

                history_embeddings = get_history_embeddings(
                    history_indices, item_embeddings
                )

                target_embeddings = history_embeddings.mean(dim=1)

                user_embeddings = model(
                    history_embeddings,
                    age,
                    gender,
                    region,
                )

                loss = criterion(user_embeddings, target_embeddings)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        pbar.set_postfix(
            {"Train Loss": f"{avg_train_loss:.4f}", "Val Loss": f"{avg_val_loss:.4f}"}
        )

    model.eval()
    test_loss = 0.0
    test_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            history_indices = batch["watch_history"].to(device)
            age = batch["age"].to(device)
            gender = batch["gender"].to(device)
            region = batch["region"].to(device)

            history_embeddings = get_history_embeddings(
                history_indices, item_embeddings
            )

            target_embeddings = history_embeddings.mean(dim=1)

            user_embeddings = model(
                history_embeddings,
                age,
                gender,
                region,
            )

            loss = criterion(user_embeddings, target_embeddings)
            test_loss += loss.item()
            test_batches += 1

    print(f"Test Loss: {test_loss / test_batches:.4f}")

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()

    plot_path = save_path.replace(".pt", "_loss.png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
