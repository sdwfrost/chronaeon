"""
test_autoclock_imports.py
-------------------------
Dedicated import test for chronaeon.autoclock, exercising the conditional
import block (lines 374-377) where aeon_core.inference / aeon_core.splits
are imported inside a try block.
"""

import importlib
import sys

import chronaeon.autoclock


def test_autoclock_module_imports():
    """Verify that chronaeon.autoclock can be imported."""
    assert chronaeon.autoclock is not None
    assert hasattr(chronaeon.autoclock, "AutoClockDeconvolution")
    assert hasattr(chronaeon.autoclock, "HierarchicalAutoClock")
    assert hasattr(chronaeon.autoclock, "run_autoclock_deconvolution")
    assert hasattr(chronaeon.autoclock, "run_hierarchical_autoclock")


def test_conditional_import_block_executed():
    """Verify that the conditional import block for aeon_core.inference / aeon_core.splits
    is reachable and does not raise when torch is available."""
    try:
        import torch
        _has_torch = True
    except ImportError:
        _has_torch = False

    if not _has_torch:
        import pytest
        pytest.skip("torch not installed")

    from aeon_core.inference import load_model, prepare_alignment, get_device
    from aeon_core.splits import extract_cross_taxa_attentions_and_embeddings

    assert load_model is not None
    assert prepare_alignment is not None
    assert get_device is not None
    assert extract_cross_taxa_attentions_and_embeddings is not None


def test_autoclock_select_adaptive_n_landmarks():
    """Verify the select_adaptive_n_landmarks function is importable and callable."""
    from chronaeon.autoclock import select_adaptive_n_landmarks

    m = select_adaptive_n_landmarks(n_taxa=100, max_k=4)
    assert m == 100
