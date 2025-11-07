"""Command-line interfaces for Detree."""

from .train import main as train_main, build_argument_parser as build_train_parser
from .embeddings import main as embeddings_main, build_argument_parser as build_embeddings_parser
from .merge_lora import main as merge_lora_main, build_argument_parser as build_merge_lora_parser
from .test_score_knn import main as test_score_knn_main, build_argument_parser as build_test_score_knn_parser
from .test_database_score_knn import main as test_database_score_knn_main, build_argument_parser as build_test_database_score_knn_parser
from .hierarchical_clustering import main as hierarchical_clustering_main, build_argument_parser as build_hierarchical_clustering_parser
from .similarity_matrix import main as similarity_matrix_main, build_argument_parser as build_similarity_matrix_parser
from .database import main as database_main, build_argument_parser as build_database_parser
from .gen_tree import main as gen_tree_main, build_argument_parser as build_gen_tree_parser

__all__ = [
    "train_main",
    "embeddings_main",
    "merge_lora_main",
    "test_score_knn_main",
    "test_database_score_knn_main",
    "hierarchical_clustering_main",
    "similarity_matrix_main",
    "database_main",
    "gen_tree_main",
    "build_train_parser",
    "build_embeddings_parser",
    "build_merge_lora_parser",
    "build_test_score_knn_parser",
    "build_test_database_score_knn_parser",
    "build_hierarchical_clustering_parser",
    "build_similarity_matrix_parser",
    "build_database_parser",
    "build_gen_tree_parser",
]
