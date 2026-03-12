import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import yaml
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data.content_dataset import ContentDataset
from model.intrinsic import IntrinsicEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="./cfgs/intrinsic.yml")
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
    test_ratio = training_cfg["test_ratio"]

    num_genres = model_cfg["num_genres"]
    num_keywords = model_cfg["num_keywords"]
    num_languages = model_cfg["num_languages"]
    num_metrics = model_cfg["num_metrics"]
    hidden_dim = model_cfg["hidden_dim"]
    embedding_dim = model_cfg["embedding_dim"]

    data_path = data_cfg["path"]
    save_path = save_cfg["path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading content dataset...")
    dataset = ContentDataset(data_path)
    total = len(dataset)

    print("Creating data loaders...")
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    print("Creating Intrisic Model...")
    model = IntrinsicEvaluator(
        num_genres=num_genres,
        num_keywords=num_keywords,
        num_languages=num_languages,
        num_metrics=num_metrics,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    train_losses = []
    val_losses = []

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(
                batch["genres"].to(device),
                batch["keywords"].to(device),
                batch["language"].to(device),
                batch["metrics_vector"].to(device),
            )
            loss = criterion(output, batch["popularity"].to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    batch["genres"].to(device),
                    batch["keywords"].to(device),
                    batch["language"].to(device),
                    batch["metrics_vector"].to(device),
                )
                val_loss += criterion(output, batch["popularity"].to(device)).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        pbar.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}"
        })

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            output = model(
                batch["genres"].to(device),
                batch["keywords"].to(device),
                batch["language"].to(device),
                batch["metrics_vector"].to(device),
            )
            test_loss += criterion(output, batch["popularity"].to(device)).item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

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
