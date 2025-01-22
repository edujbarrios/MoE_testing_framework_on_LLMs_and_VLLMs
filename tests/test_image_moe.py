import pytest
import numpy as np
from image_processing.image_moe import ImageMoE

def test_image_moe_initialization():
    """Test ImageMoE initialization."""
    moe = ImageMoE(num_experts=4)
    assert moe.num_experts == 4
    assert hasattr(moe, 'dashboard')

def test_compute_complexity():
    """Test complexity computation for different image regions."""
    moe = ImageMoE()
    
    # Test uniform region (should have low complexity)
    uniform = np.ones((8, 8)) * 0.5
    uniform_complexity = moe._compute_complexity(uniform)
    
    # Test high variance region (should have higher complexity)
    random = np.random.rand(8, 8)
    random_complexity = moe._compute_complexity(random)
    
    assert uniform_complexity < random_complexity

def test_get_region_features():
    """Test feature extraction from image regions."""
    moe = ImageMoE()
    
    region = np.random.rand(8, 8)
    features = moe._get_region_features(region)
    
    assert 'mean' in features
    assert 'std' in features
    assert 'max' in features
    assert 'min' in features

def test_process_invalid_input():
    """Test error handling for invalid inputs."""
    moe = ImageMoE()
    
    with pytest.raises(ValueError):
        moe.process(None)
    
    with pytest.raises(ValueError):
        moe.process(np.array([1, 2, 3]))  # 1D array
    
    with pytest.raises(ValueError):
        moe.process(np.random.rand(8, 8, 3))  # 3D array

def test_process_valid_input():
    """Test processing of valid image input."""
    moe = ImageMoE()
    image = np.random.rand(32, 32)
    result = moe.process(image)
    
    assert isinstance(result, str)
    assert "Expert" in result
    assert "specialist" in result
