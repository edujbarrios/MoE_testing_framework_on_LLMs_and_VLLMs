import pytest
import numpy as np
from moe_variants.switched_moe import SwitchedMoE

def test_switched_moe_initialization():
    """Test SwitchedMoE initialization."""
    moe = SwitchedMoE(num_experts=3, complexity_threshold=0.5)
    assert moe.num_experts == 3
    assert moe.complexity_threshold == 0.5

def test_compute_complexity():
    """Test complexity computation for different input types."""
    moe = SwitchedMoE()
    
    # Test text complexity
    text_simple = moe._compute_complexity("a")
    text_complex = moe._compute_complexity("supercalifragilistic")
    assert text_simple < text_complex
    
    # Test image complexity
    image_simple = moe._compute_complexity(np.ones((8, 8)) * 0.5)
    image_complex = moe._compute_complexity(np.random.rand(8, 8))
    assert image_simple < image_complex

def test_route():
    """Test routing logic."""
    moe = SwitchedMoE(complexity_threshold=0.5)
    
    # Simple input should go to expert 0
    simple_route = moe.route("a")
    assert simple_route == 0
    
    # Complex input should go to expert 2
    complex_route = moe.route("supercalifragilistic")
    assert complex_route == 2

def test_process():
    """Test processing with different inputs."""
    moe = SwitchedMoE()
    
    result = moe.process("test")
    assert isinstance(result, str)
    assert "Expert" in result
    assert "complexity" in result
    assert "conf" in result

def test_get_metrics():
    """Test metrics retrieval."""
    moe = SwitchedMoE()
    metrics = moe.get_metrics()
    
    assert 'complexity_threshold' in metrics
    assert 'expert_descriptions' in metrics
    assert len(metrics['expert_descriptions']) == 3
