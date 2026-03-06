"""Command-line interfaces for Detree.

Imports are lazy to avoid RuntimeWarning when running modules with `python -m`.
"""


def __getattr__(name: str):
    """Lazy import to prevent 'found in sys.modules' warnings."""
    _imports = {
        "train_main": (".train", "main"),
        "build_train_parser": (".train", "build_argument_parser"),
        "embeddings_main": (".embeddings", "main"),
        "build_embeddings_parser": (".embeddings", "build_argument_parser"),
        "merge_lora_main": (".merge_lora", "main"),
        "build_merge_lora_parser": (".merge_lora", "build_argument_parser"),
        "test_score_knn_main": (".test_score_knn", "main"),
        "build_test_score_knn_parser": (".test_score_knn", "build_argument_parser"),
        "test_database_score_knn_main": (".test_database_score_knn", "main"),
        "build_test_database_score_knn_parser": (".test_database_score_knn", "build_argument_parser"),
        "hierarchical_clustering_main": (".hierarchical_clustering", "main"),
        "build_hierarchical_clustering_parser": (".hierarchical_clustering", "build_argument_parser"),
        "similarity_matrix_main": (".similarity_matrix", "main"),
        "build_similarity_matrix_parser": (".similarity_matrix", "build_argument_parser"),
        "database_main": (".database", "main"),
        "build_database_parser": (".database", "build_argument_parser"),
        "gen_tree_main": (".gen_tree", "main"),
        "build_gen_tree_parser": (".gen_tree", "build_argument_parser"),
        "train_clip_projector_main": (".train_clip_projector", "main"),
        "build_train_clip_projector_parser": (".train_clip_projector", "build_argument_parser"),
        "gen_image_embeddings_main": (".gen_image_embeddings", "main"),
        "build_gen_image_embeddings_parser": (".gen_image_embeddings", "build_argument_parser"),
        "merge_databases_main": (".merge_databases", "main"),
        "build_merge_databases_parser": (".merge_databases", "build_argument_parser"),
        "test_image_score_knn_main": (".test_image_score_knn", "main"),
        "build_test_image_score_knn_parser": (".test_image_score_knn", "build_argument_parser"),
        "generalization_sweep_main": (".generalization_sweep", "main"),
        "build_generalization_sweep_parser": (".generalization_sweep", "build_argument_parser"),
    }

    if name in _imports:
        module_name, attr_name = _imports[name]
        from importlib import import_module
        module = import_module(module_name, __package__)
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "train_clip_projector_main",
    "gen_image_embeddings_main",
    "merge_databases_main",
    "build_train_parser",
    "build_embeddings_parser",
    "build_merge_lora_parser",
    "build_test_score_knn_parser",
    "build_test_database_score_knn_parser",
    "build_hierarchical_clustering_parser",
    "build_similarity_matrix_parser",
    "build_database_parser",
    "build_gen_tree_parser",
    "build_train_clip_projector_parser",
    "build_gen_image_embeddings_parser",
    "build_merge_databases_parser",
    "test_image_score_knn_main",
    "build_test_image_score_knn_parser",
    "generalization_sweep_main",
    "build_generalization_sweep_parser",
]
