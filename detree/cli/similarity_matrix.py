"""Compute similarity matrices from embedding databases."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch


def gen_data(dict_data):
    embeddings = dict_data["embeddings"]
    labels = dict_data["labels"]
    ids = dict_data["ids"]
    classes = dict_data["classes"]
    return embeddings, labels, ids, classes


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate class similarity matrices for DETree.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database", type=Path, required=True, help="Path to the embedding database (.pt).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store the similarity outputs.")
    parser.add_argument("--layers", type=int, nargs="*", default=None, help="Specific layers to export. Defaults to all.")
    return parser


def compute_similarity(database: Path, output_dir: Path, layers: Optional[Iterable[int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_emb, data_labels, data_ids, data_classes = gen_data(torch.load(database))

    if layers is None:
        layers = list(data_emb.keys())

    for layer in layers:
        center = []
        for item in data_classes:
            index = data_classes.index(item)
            now_emb = data_emb[layer][data_labels == index]
            center.append(torch.mean(now_emb, dim=0))
        center = torch.stack(center)
        similarity = center @ center.T
        similarity_np = similarity.cpu().float().numpy()

        txt_path = output_dir / f"similarity_layer_{layer}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            f.write(" ".join(data_classes) + "\n")
            for i, class_name in enumerate(data_classes):
                row = " ".join(f"{similarity_np[i, j]:.4f}" for j in range(len(data_classes)))
                f.write(f"{class_name} {row}\n")

        plt.figure(figsize=(30, 30))
        plt.imshow(similarity_np, cmap="viridis")
        plt.colorbar()
        plt.xticks(range(len(data_classes)), data_classes, rotation=45, fontsize=12)
        plt.yticks(range(len(data_classes)), data_classes, fontsize=12)
        plt.title(f"Similarity Matrix (layer {layer})", fontsize=20)
        fig_path = output_dir / f"similarity_layer_{layer}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved similarity matrix for layer {layer} to {txt_path} and {fig_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    compute_similarity(args.database, args.output_dir, args.layers)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "compute_similarity", "main"]
