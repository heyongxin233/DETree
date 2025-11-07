"""Generate clustered prototype databases from embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
import torch


class GPUKMeansClusterer:
    def __init__(self, dim: int, n_clusters: int = 500, n_iter: int = 20, n_gpu: int = 1):
        self.clus = faiss.Clustering(dim, n_clusters)
        self.clus.verbose = True
        self.clus.niter = n_iter
        self.dim = dim
        self.n_clusters = n_clusters
        self.clus.update_index = True

        res = [faiss.StandardGpuResources() for _ in range(n_gpu)]
        flat_config = []
        for i in range(n_gpu):
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = i
            flat_config.append(cfg)

        if n_gpu == 1:
            self.index = faiss.GpuIndexFlatL2(res[0], self.dim, flat_config[0])
        else:
            indexes = [faiss.GpuIndexFlatL2(res[i], self.dim, flat_config[i]) for i in range(n_gpu)]
            self.index = faiss.IndexReplicas()
            for sub_index in indexes:
                self.index.addIndex(sub_index)

    def fit(self, embeddings_np: np.ndarray) -> np.ndarray:
        self.index.reset()
        self.clus.train(embeddings_np, self.index)
        centroids = faiss.vector_float_to_array(self.clus.centroids)
        centroids = centroids.reshape(self.n_clusters, self.dim)
        return centroids


def gen_data(dict_data):
    embeddings = dict_data["embeddings"]
    labels = dict_data["labels"]
    ids = dict_data["ids"]
    classes = dict_data["classes"]
    return embeddings, labels, ids, classes


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster embeddings into prototype databases using GPU K-Means.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--database", type=Path, required=True, help="Input embedding database (.pt).")
    parser.add_argument("--output", type=Path, required=True, help="Output path for the clustered database.")
    parser.add_argument("--clusters", type=int, default=10000)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--human-class-name", type=str, default="human", help="Label representing humans in the class list.")
    return parser


def cluster_database(args: argparse.Namespace) -> None:
    data_emb, data_labels, data_ids, data_classes = gen_data(torch.load(args.database))
    human_idx = data_classes.index(args.human_class_name)
    datapos = (data_labels == human_idx).long()
    pos2cnt = {0: args.clusters, 1: args.clusters}
    pos2name = {0: ["llm"], 1: ["human"]}

    datapos_np = datapos.cpu().numpy()
    kmeans = GPUKMeansClusterer(args.dimension, n_clusters=args.clusters, n_iter=args.iterations, n_gpu=args.gpus)
    all_centers = {}
    save_labels = None
    for key in data_emb:
        now_emb = data_emb[key].float().cpu().numpy()
        all_center = []
        all_labels = []
        for pos in pos2cnt:
            pos_emb = now_emb[datapos_np == pos]
            pos_center = kmeans.fit(pos_emb)
            all_center.append(pos_center)
            all_labels.append(np.full((pos_center.shape[0],), pos))
        all_center = np.concatenate(all_center, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_center = torch.from_numpy(all_center).to(dtype=torch.bfloat16)
        all_labels = torch.from_numpy(all_labels).to(dtype=torch.long)
        all_centers[key] = all_center
        save_labels = all_labels

    save_ids = torch.arange(save_labels.shape[0], dtype=torch.long)
    classes = [None] * len(pos2name.keys())
    for pos in pos2name:
        classes[pos] = ','.join(pos2name[pos])

    emb_dict = {"embeddings": all_centers, "labels": save_labels, "ids": save_ids, "classes": classes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, args.output)
    print(f"All centers saved to: {args.output}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    cluster_database(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "cluster_database", "main"]
