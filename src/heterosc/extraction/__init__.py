"""Frozen foundation-model embedding extraction.

Each foundation model has its own heavy dependencies, so the model-specific modules (``geneformer``, ``scgpt``,
``scfoundation``, ``cellfm``) import them lazily and are meant to be run in separate environments (see ``envs/``).
They are ported from the authors' working notebooks and need the foundation model's own code and checkpoint; they are
not exercised by the unit tests. Run each on one small dataset first and compare the embedding shape with the table
in ``docs/data.md``.
"""
from .common import load_embeddings, save_embeddings

__all__ = ["save_embeddings", "load_embeddings"]
