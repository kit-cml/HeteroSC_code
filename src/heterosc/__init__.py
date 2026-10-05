"""HeteroSC: adaptive expert orchestration of heterogeneous single-cell foundation models."""
from .annotation import AnnotationConfig, AnnotationTrainer, HeteroSCClassifier, make_loaders
from .data import align_experts, load_dataset
from .knn import knn_probe
from .losses import diversity_loss, normalized_entropy, routing_entropy
from .modules import AdapterMLP, AdaptiveRouter, HeteroSCFusion
from .utils import l2_normalize, set_seed

__version__ = "0.1.0"
__all__ = [
    "AdapterMLP", "AdaptiveRouter", "HeteroSCFusion", "HeteroSCClassifier", "AnnotationConfig", "AnnotationTrainer",
    "make_loaders", "load_dataset", "align_experts", "knn_probe", "diversity_loss", "routing_entropy",
    "normalized_entropy", "l2_normalize", "set_seed",
]
# Task modules are imported on demand:  heterosc.drug_response, heterosc.perturbation, heterosc.zenodo, heterosc.extraction
