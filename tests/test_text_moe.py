import pytest
import numpy as np
from text_processing.text_moe import TextMoE

def test_text_moe_initialization():
    """Test TextMoE initialization with different number of experts."""
    moe = TextMoE(num_experts=3)
    assert moe.num_experts == 3
    assert hasattr(moe, 'dashboard')

def test_compute_complexity():
    """Test complexity computation for different inputs."""
    moe = TextMoE()
    
    # Test simple cases
    assert moe._compute_complexity("a") < moe._compute_complexity("python")
    assert moe._compute_complexity("") == 0
    
    # Test with special characters
    assert moe._compute_complexity("@#$%") > moe._compute_complexity("test")

def test_get_expert_weights():
    """Test expert weight assignment."""
    moe = TextMoE(num_experts=3)
    
    # Test short word
    weights = moe._get_expert_weights("the")
    assert np.argmax(weights) == 0  # Should be assigned to expert 0 (short words)
    
    # Test medium word
    weights = moe._get_expert_weights("python")
    assert np.argmax(weights) == 1  # Should be assigned to expert 1 (medium words)
    
    # Test long word
    weights = moe._get_expert_weights("programming")
    assert np.argmax(weights) == 2  # Should be assigned to expert 2 (long words)

def test_process_invalid_input():
    """Test error handling for invalid inputs."""
    moe = TextMoE()
    
    with pytest.raises(ValueError):
        moe.process("")
    
    with pytest.raises(ValueError):
        moe.process(None)

def test_process_valid_input():
    """Test processing of valid text input."""
    moe = TextMoE()
    result = moe.process("The quick brown fox")
    
    assert isinstance(result, str)
    assert "Expert" in result
    assert "specialist" in result
